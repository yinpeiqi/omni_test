import json
import argparse
import sys
import os

def analyze_performance(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

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
    time_components = {}
    
    total_input_tokens = 0
    total_text_output_tokens = 0
    total_audio_output_tokens = 0
    
    # Throughput lists to calculate average throughput per sample
    text_throughputs = []
    audio_throughputs = []

    max_sample = 10
    data = data[:max_sample]
    count = len(data)

    for sample in data:
        stats = sample.get('stats', {})
        time_stats = stats.get('time', {})
        token_stats = stats.get('tokens', {})

        # JCT
        jct = time_stats.get('total', 0)
        total_jct += jct

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

        # Throughput per sample (avoid division by zero)
        if jct > 0:
            text_throughputs.append(text_out / jct)
            audio_throughputs.append(audio_out / jct)
        else:
            text_throughputs.append(0)
            audio_throughputs.append(0)

    # Averages
    avg_jct = total_jct / count
    avg_time_components = {k: v / count for k, v in time_components.items()}
    avg_input_tokens = total_input_tokens / count
    avg_text_output_tokens = total_text_output_tokens / count
    avg_audio_output_tokens = total_audio_output_tokens / count
    
    # Average Throughput (Average of rates)
    avg_text_throughput = sum(text_throughputs) / len(text_throughputs) if text_throughputs else 0
    avg_audio_throughput = sum(audio_throughputs) / len(audio_throughputs) if audio_throughputs else 0

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

    print("\nThroughput (Average per sample):")
    print(f"  Text: {avg_text_throughput:.2f} tokens/s ({avg_text_throughput*60:.2f} tokens/min)")
    print(f"  Audio: {avg_audio_throughput:.2f} tokens/s ({avg_audio_throughput*60:.2f} tokens/min)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <json_file_path>")
        sys.exit(1)
    
    analyze_performance(sys.argv[1])

