import json
import argparse
import sys
import os
import wave
import contextlib
import glob

def analyze_performance_vllm(results_dir, output_dir=None):
    if not os.path.exists(results_dir):
        print(f"Error: Directory not found at {results_dir}")
        return

    # Find vllm_stats.json
    stats_file = os.path.join(results_dir, "vllm_stats.json")
    if not os.path.exists(stats_file):
        print(f"Error: vllm_stats.json not found in {results_dir}")
        return

    # Find the main results JSON (excluding vllm_stats.json)
    json_files = [f for f in glob.glob(os.path.join(results_dir, "*.json")) if os.path.basename(f) != "vllm_stats.json"]
    if not json_files:
        print(f"Error: No main result JSON file found in {results_dir}")
        return
    # Use the first one found, similar to original script logic
    main_result_file = json_files[0]
    print(f"Found main result file: {main_result_file}")
    print(f"Found stats file: {stats_file}")

    if output_dir is None:
        output_dir = os.path.join(results_dir, "outputs")
    
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory not found at {output_dir}. RTF calculation might be incomplete.")

    # Read Main Result File for Total Time
    try:
        with open(main_result_file, 'r') as f:
            main_data = json.load(f)
    except Exception as e:
        print(f"Error reading main result file: {e}")
        return

    if not main_data:
        print("No data in main result file.")
        return

    # Get total batch time from the first sample
    # The user said: "这个文件读入第一个sample的total time batch, 作为所有batch的总时间"
    total_time_batch = main_data[0].get("total_time_batch", 0)
    print(f"Total Batch Time (from sample 1): {total_time_batch:.4f} s")

    # Read Stats File for Tokens and Stages
    try:
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
    except Exception as e:
        print(f"Error reading stats file: {e}")
        return

    total_tokens = 0
    stage_stats = {} # stage_id -> {'time': sum_time, 'tokens': sum_tokens, 'count': count}
    
    # Process stats data
    # Keys are like "0_uuid", "1_uuid"
    sample_count = len(stats_data)
    
    for key, sample_stats in stats_data.items():
        stages = sample_stats.get("stages", {})
        for stage_id, stage_info in stages.items():
            s_time = stage_info.get("stage_gen_time_ms", 0)
            s_tokens = stage_info.get("num_tokens_out", 0)
            
            total_tokens += s_tokens
            
            if stage_id not in stage_stats:
                stage_stats[stage_id] = {'time': 0.0, 'tokens': 0, 'count': 0}
            
            stage_stats[stage_id]['time'] += s_time
            stage_stats[stage_id]['tokens'] += s_tokens
            stage_stats[stage_id]['count'] += 1

    # Calculate Audio Duration from wav files
    total_audio_duration = 0
    
    # We iterate through main_data to match sample_ids to wav files
    # main_data has "sample_id"
    for sample in main_data:
        sample_id = sample.get('sample_id')
        if sample_id is not None and os.path.exists(output_dir):
            wav_filename = f"sample_{sample_id}_audio.wav"
            wav_path = os.path.join(output_dir, wav_filename)
            
            if os.path.exists(wav_path):
                try:
                    with contextlib.closing(wave.open(wav_path, 'r')) as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        total_audio_duration += duration
                except Exception as e:
                    print(f"Warning: Could not read audio file {wav_path}: {e}")

    # Metrics
    # Throughput = Total Tokens / Total Batch Time
    # throughput = total_tokens / total_time_batch if total_time_batch > 0 else 0
    
    # RTF = Total Batch Time / Total Audio Duration
    rtf = total_time_batch / total_audio_duration if total_audio_duration > 0 else 0

    print("-" * 50)
    print(f"Analysis for {results_dir}")
    print(f"Number of Samples: {sample_count}")
    print(f"Total Batch Time: {total_time_batch:.4f} s")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Audio Duration: {total_audio_duration:.4f} s")
    
    # print("\nThroughput:")
    # print(f"  {throughput:.2f} tokens/s")
    
    if total_audio_duration > 0:
        print("\nRTF Analysis:")
        print(f"  RTF (Total Batch Time / Total Audio Duration): {rtf:.4f}")
    
    print("\nPer-Stage Statistics (Average per sample):")
    for stage_id in sorted(stage_stats.keys(), key=lambda x: int(x) if x.isdigit() else x):
        info = stage_stats[stage_id]
        count = info['count']
        avg_time_ms = info['time'] / count
        avg_tokens = info['tokens'] / count
        
        # Calculate TPS for this stage
        total_stage_time_ms = info['time']
        total_stage_tokens = info['tokens']
        stage_tps = (total_stage_tokens / (total_stage_time_ms / 1000)) if total_stage_time_ms > 0 else 0

        stage_name = f"Stage {stage_id}"
        if str(stage_id) == "0":
            stage_name += " (Thinker)"
        elif str(stage_id) == "1":
            stage_name += " (Talker)"

        print(f"  {stage_name}:")
        print(f"    Avg Time: {avg_time_ms:.2f} ms ({avg_time_ms/1000:.4f} s)")
        print(f"    Avg Tokens: {avg_tokens:.2f}")
        print(f"    TPS: {stage_tps:.2f} tokens/s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance_vllm.py <results_dir> [<output_dir>]")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_performance_vllm(results_dir, output_dir)

