#!/usr/bin/env python3
"""
Unified Workload Test for Qwen3-Omni (vLLM)
Combines Audio, Video, and Visual tests using vllm-omni.
"""
import os
import json
import sys
import time
import argparse
import soundfile as sf
import numpy as np
import torch
import random
from PIL import Image
from transformers import set_seed

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Try to import librosa and decord
try:
    import librosa
except ImportError:
    print("Warning: librosa not found. Audio test may fail.")

try:
    import decord
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not found. Video test may fail.")

def get_omni_model(model_path, workspace_dir, args):
    from vllm_omni.entrypoints.omni import Omni

    # Determine log path based on which test is enabled
    if args.audio:
        log_subdir = "audio_test"
    elif args.image:
        log_subdir = "visual_test"
    elif args.video:
        log_subdir = "video_test"
    else:
        log_subdir = "audio_test"  # fallback
    
    log_file = os.path.join(workspace_dir, f"{log_subdir}/results_vllm/vllm_stats")
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # We need the config path. In test_audio_vllm.py it assumes it is in the same dir as the script.
    # We will assume this script is in omni_test/
    if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        config_path = os.path.join(workspace_dir, "qwen3_omni_moe.yaml")
    elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
        config_path = os.path.join(workspace_dir, "qwen2_5_omni.yaml")
    else:
        raise ValueError(f"Unsupported model path: {args.model_path}")

    print(f"Initializing Omni model from {model_path}...")
    omni_llm = Omni(
        model=model_path,
        stage_configs_path=config_path,
        log_stats=True,
    )
    return omni_llm

def get_sampling_params():
    from vllm import SamplingParams
    SEED = 42
    
    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1200,
        repetition_penalty=1.05,
        logit_bias={},
        seed=SEED,
    )

    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[2150],  # TALKER_CODEC_EOS_TOKEN_ID
    )

    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    return [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

def load_video_frames(video_path, max_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= max_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    # Frames are (T, H, W, C), need to be compatible with vLLM
    # vLLM usually expects list of PIL images or numpy array
    return [Image.fromarray(frame) for frame in frames]

def run_audio_test_vllm(omni_llm, sampling_params_list, num_samples, workspace_dir, args):
    from datasets import load_dataset
    sys.path.append(workspace_dir)
    from prompts import DEFAULT_AUDIO_PROMPT

    print("\n" + "=" * 60)
    print("Running Audio Workload Test (vLLM)")
    print("Using Prompt: ", DEFAULT_AUDIO_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "audio_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, "results_vllm")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading LibriSpeech dataset (first {num_samples} samples)...")
    dataset = load_dataset("librispeech_asr", "clean", split=f"test[:{num_samples}]", trust_remote_code=True)
    
    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )
    
    prompts = []
    sample_ids = []
    ground_truths = []
    
    print("Preparing inputs...")
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        
        # Save original audio
        audio_path = os.path.join(data_dir, f"sample_{i+1}.wav")
        if not os.path.exists(audio_path):
            sf.write(audio_path, audio_array, sample_rate)
            
        # Load audio for vllm (resample to 16k)
        y, sr = librosa.load(audio_path, sr=16000)
        audio_data = (y.astype(np.float32), sr)
        
        # Construct prompt
        prompt_text = DEFAULT_AUDIO_PROMPT
        if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        
        inputs = {
            "prompt": prompt_str,
            "multi_modal_data": {
                "audio": audio_data,
            },
        }
        prompts.append(inputs)
        sample_ids.append(i+1)
        if "text" in sample:
            ground_truths.append(sample["text"])
        else:
            ground_truths.append("")
            
    print(f"Prepared {len(prompts)} prompts.")
    
    print("\nGenerating responses (Batch)...")
    start_time = time.time()
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    total_time = time.time() - start_time
    print(f"Generation completed in {total_time:.2f}s")
    
    results = []
    for idx, s_id in enumerate(sample_ids):
        results.append({
            "sample_id": s_id,
            "prompt": DEFAULT_AUDIO_PROMPT,
            "text_output": "",
            "has_audio_output": False,
            "ground_truth": ground_truths[idx]
        })
        
    avg_time = total_time / len(prompts)
    
    print("Processing outputs...")
    for stage_outputs in omni_outputs:
        output_type = stage_outputs.final_output_type
        
        for output in stage_outputs.request_output:
            req_id = int(output.request_id.split("_")[0])
            if req_id >= len(results):
                continue
            
            result = results[req_id]
            
            if output_type == "text":
                text_output = output.outputs[0].text
                result["text_output"] = text_output
                
                # Save text
                text_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_text.txt")
                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {DEFAULT_AUDIO_PROMPT}\n\n")
                    if result.get("ground_truth"):
                        f.write(f"Ground Truth: {result['ground_truth']}\n\n")
                    f.write(f"Response: {text_output}\n")
                    
            elif output_type == "audio":
                if "audio" in output.multimodal_output:
                    audio_tensor = output.multimodal_output["audio"]
                    audio_numpy = audio_tensor.float().detach().cpu().numpy()
                    if audio_numpy.ndim > 1:
                        audio_numpy = audio_numpy.flatten()
                        
                    audio_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_audio.wav")
                    sf.write(audio_output_path, audio_numpy, samplerate=24000, format="WAV")
                    result["has_audio_output"] = True

    # Save results
    for r in results:
        r["output_decode_time"] = avg_time
        r["total_time_batch"] = total_time

    if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        result_json_path = os.path.join(results_dir, "Qwen3-Omni_vllm_audio.json")
    elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
        result_json_path = os.path.join(results_dir, "Qwen2.5-Omni_vllm_audio.json")

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_json_path}")

def run_video_test_vllm(omni_llm, sampling_params_list, num_samples, workspace_dir, args):
    from datasets import load_dataset
    try:
        sys.path.append(workspace_dir)
        from prompts import DEFAULT_VIDEO_PROMPT
    except ImportError:
        DEFAULT_VIDEO_PROMPT = "Describe the video."

    print("\n" + "=" * 60)
    print("Running Video Workload Test (vLLM)")
    print("Using Prompt: ", DEFAULT_VIDEO_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "video_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, "results_vllm")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading ucf101-subset dataset (first {num_samples} samples)...")
    dataset = load_dataset("sayakpaul/ucf101-subset", split=f"train[:{num_samples}]", trust_remote_code=True)
    
    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )
    
    prompts = []
    sample_ids = []
    reference_labels = []

    print("Preparing inputs...")
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        video_bytes = sample["avi"]
        video_path = os.path.join(data_dir, f"sample_{i+1}.avi")
        if not os.path.exists(video_path):
            with open(video_path, "wb") as f:
                f.write(video_bytes)
        
        # Load video frames
        # For vLLM Qwen2-VL, typically input is list of frames or tensor
        # We'll use list of PIL images if possible, or whatever vllm_omni supports
        # Assuming standard vLLM Image/Video inputs
        try:
            frames = load_video_frames(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            continue

        prompt_text = DEFAULT_VIDEO_PROMPT
        if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        
        inputs = {
            "prompt": prompt_str,
            "multi_modal_data": {
                "video": frames, # Pass list of PIL images
            },
        }
        prompts.append(inputs)
        sample_ids.append(i+1)
        
        label_text = ""
        if "label" in sample:
            try:
                label_int = sample['label']
                if hasattr(dataset.features['label'], 'int2str'):
                    label_text = dataset.features['label'].int2str(label_int)
                else:
                    label_text = str(label_int)
            except:
                label_text = str(sample.get('label', ''))
        reference_labels.append(label_text)

    print(f"Prepared {len(prompts)} prompts.")
    
    print("\nGenerating responses (Batch)...")
    start_time = time.time()
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    total_time = time.time() - start_time
    print(f"Generation completed in {total_time:.2f}s")
    
    results = []
    for idx, s_id in enumerate(sample_ids):
        results.append({
            "sample_id": s_id,
            "prompt": DEFAULT_VIDEO_PROMPT,
            "text_output": "",
            "has_audio_output": False,
            "reference_label": reference_labels[idx]
        })
        
    avg_time = total_time / len(prompts) if prompts else 0
    
    print("Processing outputs...")
    for stage_outputs in omni_outputs:
        output_type = stage_outputs.final_output_type
        
        for output in stage_outputs.request_output:
            req_id = int(output.request_id.split("_")[0])
            if req_id >= len(results):
                continue
            
            result = results[req_id]
            
            if output_type == "text":
                text_output = output.outputs[0].text
                result["text_output"] = text_output
                
                # Save text
                text_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_text.txt")
                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {DEFAULT_VIDEO_PROMPT}\n\n")
                    if result.get("reference_label"):
                        f.write(f"Reference Label: {result['reference_label']}\n\n")
                    f.write(f"Response: {text_output}\n")
                    
            elif output_type == "audio":
                if "audio" in output.multimodal_output:
                    audio_tensor = output.multimodal_output["audio"]
                    audio_numpy = audio_tensor.float().detach().cpu().numpy()
                    if audio_numpy.ndim > 1:
                        audio_numpy = audio_numpy.flatten()
                        
                    audio_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_audio.wav")
                    sf.write(audio_output_path, audio_numpy, samplerate=24000, format="WAV")
                    result["has_audio_output"] = True

    # Save results
    for r in results:
        r["output_decode_time"] = avg_time
        r["total_time_batch"] = total_time

    
    if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        result_json_path = os.path.join(results_dir, "Qwen3-Omni_vllm_video.json")
    elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
        result_json_path = os.path.join(results_dir, "Qwen2.5-Omni_vllm_video.json")

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_json_path}")

def run_visual_test_vllm(omni_llm, sampling_params_list, num_samples, workspace_dir, args):
    from datasets import load_dataset
    sys.path.append(workspace_dir)
    from prompts import DEFAULT_VISUAL_PROMPT

    print("\n" + "=" * 60)
    print("Running Visual Workload Test (vLLM)")
    print("Using Prompt: ", DEFAULT_VISUAL_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "visual_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, "results_vllm")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading food101 dataset (first {num_samples} samples)...")
    dataset = load_dataset("food101", split=f"validation[:{num_samples}]")
    
    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )
    
    prompts = []
    sample_ids = []
    reference_captions = []

    print("Preparing inputs...")
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        image = sample["image"]
        image_path = os.path.join(data_dir, f"sample_{i+1}.jpg")
        if not os.path.exists(image_path):
            image.save(image_path)
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt_text = DEFAULT_VISUAL_PROMPT
        if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
            prompt_str = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>"
                f"{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        
        inputs = {
            "prompt": prompt_str,
            "multi_modal_data": {
                "image": image,
            },
        }
        prompts.append(inputs)
        sample_ids.append(i+1)
        
        reference_captions.append(str(sample.get('label', '')))

    print(f"Prepared {len(prompts)} prompts.")
    
    print("\nGenerating responses (Batch)...")
    start_time = time.time()
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    total_time = time.time() - start_time
    print(f"Generation completed in {total_time:.2f}s")
    
    results = []
    for idx, s_id in enumerate(sample_ids):
        results.append({
            "sample_id": s_id,
            "prompt": DEFAULT_VISUAL_PROMPT,
            "text_output": "",
            "has_audio_output": False,
            "reference_caption": reference_captions[idx]
        })
        
    avg_time = total_time / len(prompts) if prompts else 0
    
    print("Processing outputs...")
    for stage_outputs in omni_outputs:
        output_type = stage_outputs.final_output_type
        
        for output in stage_outputs.request_output:
            req_id = int(output.request_id.split("_")[0])
            if req_id >= len(results):
                continue
            
            result = results[req_id]
            
            if output_type == "text":
                text_output = output.outputs[0].text
                result["text_output"] = text_output
                
                # Save text
                text_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_text.txt")
                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {DEFAULT_VISUAL_PROMPT}\n\n")
                    if result.get("reference_caption"):
                        f.write(f"Reference Caption: {result['reference_caption']}\n\n")
                    f.write(f"Response: {text_output}\n")
                    
            elif output_type == "audio":
                if "audio" in output.multimodal_output:
                    audio_tensor = output.multimodal_output["audio"]
                    audio_numpy = audio_tensor.float().detach().cpu().numpy()
                    if audio_numpy.ndim > 1:
                        audio_numpy = audio_numpy.flatten()
                        
                    audio_output_path = os.path.join(output_dir, f"sample_{result['sample_id']}_audio.wav")
                    sf.write(audio_output_path, audio_numpy, samplerate=24000, format="WAV")
                    result["has_audio_output"] = True

    # Save results
    for r in results:
        r["output_decode_time"] = avg_time
        r["total_time_batch"] = total_time

    if args.model_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        result_json_path = os.path.join(results_dir, "Qwen3-Omni_vllm_image.json")
    elif args.model_path == "Qwen/Qwen2.5-Omni-7B":
        result_json_path = os.path.join(results_dir, "Qwen2.5-Omni_vllm_image.json")

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni Unified Workload Test (vLLM)")
    parser.add_argument("--audio", action="store_true", help="Run audio test")
    parser.add_argument("--video", action="store_true", help="Run video test")
    parser.add_argument("--image", action="store_true", help="Run visual test")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Path to model")
    
    args = parser.parse_args()
    
    if not (args.audio or args.video or args.image):
        print("Please specify at least one test mode: --audio, --video, or --image")
        return

    seed_everything(42)
    workspace_dir = os.getcwd()
    
    # Initialize Omni Model
    omni_llm = get_omni_model(args.model_path, workspace_dir, args)
    sampling_params_list = get_sampling_params()
    
    if args.audio:
        run_audio_test_vllm(omni_llm, sampling_params_list, args.num_samples, workspace_dir, args)
        
    if args.video:
        run_video_test_vllm(omni_llm, sampling_params_list, args.num_samples, workspace_dir, args)
        
    if args.image:
        run_visual_test_vllm(omni_llm, sampling_params_list, args.num_samples, workspace_dir, args)

    # Determine log path based on which test is enabled
    if args.audio:
        log_subdir = "audio_test"
    elif args.image:
        log_subdir = "visual_test"
    elif args.video:
        log_subdir = "video_test"
    
    log_file = os.path.join(workspace_dir, f"{log_subdir}/results_vllm/vllm_stats.json")
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(omni_llm.metrics.per_request, f, indent=2, ensure_ascii=False)
    print("\nAll requested tests completed!")
    omni_llm.close()

if __name__ == "__main__":
    main()

