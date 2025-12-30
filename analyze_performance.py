import json
import argparse
import sys
import os
import wave
import contextlib

def analyze_performance(file_path, output_dir=None):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    if os.path.isdir(file_path):
        base_dir = file_path
        # If input is a directory, look for a JSON file inside
        json_files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
        if not json_files:
            print(f"Error: No JSON file found in directory {base_dir}")
            return
        # Use the first JSON file found (or prioritize if needed)
        # If there are multiple, we might want to warn or pick specific ones. 
        # For now, picking the first one.
        file_path = os.path.join(base_dir, json_files[0])
        print(f"Found JSON file: {file_path}")
        
        if output_dir is None:
            output_dir = os.path.join(base_dir, "outputs")
    
    # Determine output directory (if file_path was a file originally)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), "outputs")
    
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory not found at {output_dir}. RTF calculation will be skipped.")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not data:
        print("No data found in file.")
        return

    total_jct = 0
    total_thinker_time = 0
    total_talker_time = 0
    time_components = {}
    
    total_input_tokens = 0
    total_text_output_tokens = 0
    total_audio_output_tokens = 0
    
    # RTF
    total_audio_duration = 0
    total_rtf_jct = 0
    
    # max_sample = 10
    # data = data[:max_sample]
    count = len(data)

    for sample in data:
        stats = sample.get('stats', {})
        time_stats = stats.get('time', {})
        token_stats = stats.get('tokens', {})

        # JCT
        jct = time_stats.get('total', 0)
        total_jct += jct

        thinker_time = time_stats.get('thinker_generation', 0)
        total_thinker_time += thinker_time
        talker_time = time_stats.get('talker_generation', 0)
        total_talker_time += talker_time

        # Time Components
        for k, v in time_stats.items():
            if k == 'total': continue
            time_components[k] = time_components.get(k, 0) + v

        # Tokens
        thinker_input_text = token_stats.get('thinker_input_text', 0)
        thinker_input_audio = token_stats.get('thinker_input_audio', 0)
        thinker_input_image = token_stats.get('thinker_input_image', 0)
        # Note: video input tokens might be implicitly part of image tokens in some pipelines or separate
        # Looking at result JSON, there is talker_input_video, but check if thinker has it too.
        # Based on search result: "talker_input_video": 420.
        # Let's check for thinker_input_video as well, though search result didn't explicitly show it in thinker_input_*
        thinker_input_video = token_stats.get('thinker_input_video', 0)
        
        input_tokens = thinker_input_text + thinker_input_audio + thinker_input_image + thinker_input_video
        total_input_tokens += input_tokens

        text_out = token_stats.get('thinker_output_text', 0)
        total_text_output_tokens += text_out

        audio_out = token_stats.get('talker_output_tokens', 0)
        total_audio_output_tokens += audio_out

        # RTF Calculation
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
                        total_rtf_jct += jct
                except Exception as e:
                    print(f"Warning: Could not read audio file {wav_path}: {e}")

    # Averages
    avg_jct = total_jct / count
    avg_time_components = {k: v / count for k, v in time_components.items()}
    avg_input_tokens = total_input_tokens / count
    avg_text_output_tokens = total_text_output_tokens / count
    avg_audio_output_tokens = total_audio_output_tokens / count
    
    # Average Throughput (Total Tokens / Total JCT)
    # avg_text_throughput = total_text_output_tokens / total_jct if total_jct > 0 else 0
    # avg_audio_throughput = total_audio_output_tokens / total_jct if total_jct > 0 else 0
    
    thinker_tps = total_text_output_tokens / total_thinker_time if total_thinker_time > 0 else 0
    talker_tps = total_audio_output_tokens / total_talker_time if total_talker_time > 0 else 0

    print(f"Analysis for {file_path}")
    print("-" * 50)
    print(f"Number of Samples: {count}")
    print(f"Average JCT: {avg_jct:.4f} s")
    print("\nAverage Time Components:")
    # Sort keys for consistent output
    for k in sorted(avg_time_components.keys()):
        print(f"  {k}: {avg_time_components[k]:.4f} s")
    
    print("\nToken Statistics (Average):")
    print(f"  Input Tokens: {avg_input_tokens:.2f}")
    # Additional breakdown for input
    avg_video_tokens = sum(s.get('stats', {}).get('tokens', {}).get('thinker_input_video', 0) for s in data) / count
    # Check talker tokens too as seen in the example json
    avg_talker_video_tokens = sum(s.get('stats', {}).get('tokens', {}).get('talker_input_video', 0) for s in data) / count
    
    if avg_video_tokens > 0:
        print(f"    - Thinker Video: {avg_video_tokens:.2f}")
    if avg_talker_video_tokens > 0:
        print(f"    - Talker Video: {avg_talker_video_tokens:.2f}")

    print(f"  Thinker Output Text Tokens: {avg_text_output_tokens:.2f}")
    print(f"  Talker Output Audio Tokens: {avg_audio_output_tokens:.2f}")

    print("\nThroughput (Tokens / Execution Time):")
    print(f"  Thinker TPS: {thinker_tps:.2f} tokens/s")
    print(f"  Talker TPS: {talker_tps:.2f} tokens/s")

    if total_audio_duration > 0:
        avg_rtf = total_rtf_jct / total_audio_duration
        print("\nRTF Analysis:")
        print(f"  Total Audio Duration: {total_audio_duration:.4f} s")
        print(f"  Average RTF (Total JCT / Total Duration): {avg_rtf:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <json_file_path> [<output_dir>]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_performance(file_path, output_dir)

