#!/usr/bin/env python3
"""
Unified Workload Test for Qwen3-Omni (Transformers)
Combines Audio, Video, and Visual tests.
"""
import os
import json
import sys
import time
import argparse
import soundfile as sf
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import Qwen3OmniMoeProcessor, set_seed
from transformers_qwen3_omni import Qwen3OmniMoeForConditionalGenerationWithLogging
from qwen_omni_utils import process_mm_info
from prompts import DEFAULT_AUDIO_PROMPT, DEFAULT_VIDEO_PROMPT, DEFAULT_VISUAL_PROMPT

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_and_save_sample(
    model, 
    processor, 
    sample_id, 
    prompt, 
    conversation, 
    output_dir, 
    result_json_path, 
    results_list,
    reference_data=None  # dict mapping label -> content
):
    # Process inputs
    time_input_start = time.time()
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)
    input_process_time = time.time() - time_input_start

    print("Generating response...")
    generate_result = model.generate(
        **inputs,
        speaker="Ethan",
        thinker_return_dict_in_generate=True,
        use_audio_in_video=False,
        do_sample=False,
        temperature=0.0,
    )
    
    text_ids, audio_output, stats = generate_result
    
    # Decode
    time_decode_start = time.time()
    text_output = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    output_decode_time = time.time() - time_decode_start
    
    print(f"Text output: {text_output}")
    
    # Save Text Output
    text_output_path = os.path.join(output_dir, f"sample_{sample_id}_text.txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\n")
        if reference_data:
            for label, value in reference_data.items():
                f.write(f"{label}: {value}\n\n")
        f.write(f"Response: {text_output}\n")
        
    # Save Audio Output
    if audio_output is not None:
        audio_output_path = os.path.join(output_dir, f"sample_{sample_id}_audio.wav")
        sf.write(audio_output_path, audio_output.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    
    # Update Stats
    if stats:
        if "time" not in stats: stats["time"] = {}
        stats["time"]["input_process"] = input_process_time
        stats["time"]["output_decode"] = output_decode_time
        
    # Construct Result
    result = {
        "sample_id": sample_id,
        "prompt": prompt,
        "text_output": text_output,
        "has_audio_output": audio_output is not None,
        "input_process_time": input_process_time,
        "output_decode_time": output_decode_time,
        "stats": stats if stats else None
    }
    
    if reference_data:
        # Convert keys like "Ground Truth" -> "ground_truth" for JSON
        for k, v in reference_data.items():
             key = k.lower().replace(" ", "_")
             result[key] = v

    results_list.append(result)
    
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

def run_audio_test(model, processor, num_samples, workspace_dir, gpu_count):
    print("\n" + "=" * 60)
    print("Running Audio Workload Test")
    print("Using Prompt: ", DEFAULT_AUDIO_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "audio_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, f"results_xfmrs_{gpu_count}gpu")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading LibriSpeech dataset (first {num_samples} samples)...")
    dataset = load_dataset("librispeech_asr", "clean", split=f"test[:{num_samples}]", trust_remote_code=True)
    
    results = []
    result_json_path = os.path.join(results_dir, "Qwen3-Omni_transformers_audio.json")
    
    # Pre-save audio
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        audio_path = os.path.join(data_dir, f"sample_{i+1}.wav")
        if not os.path.exists(audio_path):
            sf.write(audio_path, audio_array, sample_rate)
            print(f"Saved audio: {audio_path}")
            
    for i in range(min(len(dataset), num_samples)):
        print(f"\nProcessing Audio Sample {i+1}/{min(len(dataset), num_samples)}")
        sample = dataset[i]
        audio_path = os.path.join(data_dir, f"sample_{i+1}.wav")
        prompt = DEFAULT_AUDIO_PROMPT
        
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        
        reference_data = {}
        if "text" in sample:
            reference_data["Ground Truth"] = sample["text"]

        process_and_save_sample(
            model=model,
            processor=processor,
            sample_id=i+1,
            prompt=prompt,
            conversation=conversation,
            output_dir=output_dir,
            result_json_path=result_json_path,
            results_list=results,
            reference_data=reference_data
        )

def run_video_test(model, processor, num_samples, workspace_dir, gpu_count):
    print("\n" + "=" * 60)
    print("Running Video Workload Test")
    print("Using Prompt: ", DEFAULT_VIDEO_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "video_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, f"results_xfmrs_{gpu_count}gpu")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading ucf101-subset dataset (first {num_samples} samples)...")
    dataset = load_dataset("sayakpaul/ucf101-subset", split=f"train[:{num_samples}]", trust_remote_code=True)
    
    results = []
    result_json_path = os.path.join(results_dir, "Qwen3-Omni_transformers_video.json")
    
    # Pre-save video
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        video_bytes = sample["avi"]
        video_path = os.path.join(data_dir, f"sample_{i+1}.avi")
        if not os.path.exists(video_path):
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            print(f"Saved video: {video_path}")
            
    for i in range(min(len(dataset), num_samples)):
        print(f"\nProcessing Video Sample {i+1}/{min(len(dataset), num_samples)}")
        sample = dataset[i]
        video_path = os.path.join(data_dir, f"sample_{i+1}.avi")
        prompt = DEFAULT_VIDEO_PROMPT
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        
        reference_data = {}
        if "label" in sample:
            try:
                label_int = sample['label']
                if hasattr(dataset.features['label'], 'int2str'):
                    reference_caption = dataset.features['label'].int2str(label_int)
                else:
                    reference_caption = str(label_int)
            except:
                reference_caption = str(sample.get('label', ''))
            reference_data["Reference Label"] = reference_caption

        process_and_save_sample(
            model=model,
            processor=processor,
            sample_id=i+1,
            prompt=prompt,
            conversation=conversation,
            output_dir=output_dir,
            result_json_path=result_json_path,
            results_list=results,
            reference_data=reference_data
        )

def run_visual_test(model, processor, num_samples, workspace_dir, gpu_count):
    print("\n" + "=" * 60)
    print("Running Visual Workload Test")
    print("Using Prompt: ", DEFAULT_VISUAL_PROMPT)
    print("=" * 60)
    
    test_dir = os.path.join(workspace_dir, "visual_test")
    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, f"results_xfmrs_{gpu_count}gpu")
    output_dir = os.path.join(results_dir, "outputs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading food101 dataset (first {num_samples} samples)...")
    dataset = load_dataset("food101", split=f"validation[:{num_samples}]")
    
    results = []
    result_json_path = os.path.join(results_dir, "Qwen3-Omni_transformers_image.json")
    
    # Pre-save images
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        image = sample["image"]
        image_path = os.path.join(data_dir, f"sample_{i+1}.jpg")
        if not os.path.exists(image_path):
            image.save(image_path)
            print(f"Saved image: {image_path}")
            
    for i in range(min(len(dataset), num_samples)):
        print(f"\nProcessing Visual Sample {i+1}/{min(len(dataset), num_samples)}")
        sample = dataset[i]
        image_path = os.path.join(data_dir, f"sample_{i+1}.jpg")
        prompt = DEFAULT_VISUAL_PROMPT
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        
        reference_data = {}
        if "label" in sample:
            reference_data["Reference Caption"] = str(sample['label'])

        process_and_save_sample(
            model=model,
            processor=processor,
            sample_id=i+1,
            prompt=prompt,
            conversation=conversation,
            output_dir=output_dir,
            result_json_path=result_json_path,
            results_list=results,
            reference_data=reference_data
        )

def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni Unified Workload Test")
    parser.add_argument("--audio", action="store_true", help="Run audio test")
    parser.add_argument("--video", action="store_true", help="Run video test")
    parser.add_argument("--image", action="store_true", help="Run visual test")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Path to model")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.gpu)))
    
    if not (args.audio or args.video or args.image):
        print("Please specify at least one test mode: --audio, --video, or --image")
        return

    seed_everything(42)
    workspace_dir = os.getcwd()
    
    print("\nLoading model and processor...")
    model = Qwen3OmniMoeForConditionalGenerationWithLogging.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto" if args.gpu > 1 else "cuda:0",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print("Model loaded successfully!")
    
    if args.audio:
        run_audio_test(model, processor, args.num_samples, workspace_dir, args.gpu)
        
    if args.video:
        run_video_test(model, processor, args.num_samples, workspace_dir, args.gpu)
        
    if args.image:
        run_visual_test(model, processor, args.num_samples, workspace_dir, args.gpu)

    print("\nAll requested tests completed!")

if __name__ == "__main__":
    main()
