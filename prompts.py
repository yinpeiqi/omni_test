# Standard prompts from Qwen3-Omni cookbooks

# Audio Recognition (ASR)
AUDIO_RECOGNITION_PROMPTS = {
    "english": "Transcribe this audio.",
    "chinese": "请将这段中文语音转换为纯文本。",
    "french": "Transcribe the French audio into text.",
    "generic": "Transcribe this audio.",
}

# Audio Caption
AUDIO_CAPTION_PROMPTS = {
    "detailed": "Give the detailed description of the audio.",
    "thorough": "Give a thorough description of the audio.",
    "please": "Please provide a detailed description of the audio.",
}

# Speech Translation
AUDIO_TRANSLATION_PROMPTS = {
    "chinese_to_english": "Listen to the provided Chinese speech and produce a translation in English text.",
    "english_to_chinese": "Listen to the provided English speech and produce a translation in Chinese text.",
    "french_to_english": "Listen to the provided French speech and produce a translation in English text.",
    "generic": "Translate this audio to {target_language}.",
}

# Sound Analysis
AUDIO_ANALYSIS_PROMPTS = {
    "what_happened": "What happened in the audio?",
    "what_sound": "What is this sound? In what kind of situation might it occur?",
    "where": "Guess where I am?",
}

# Music Analysis
AUDIO_MUSIC_ANALYSIS_PROMPTS = {
    "detailed": "Describe the style, rhythm, dynamics, and expressed emotions of this piece of music. Identify the instruments used and suggest possible scenarios from which this music might originate.",
}

# Visual Description
VISUAL_DESCRIPTION_PROMPTS = {
    "simple": "Describe this image. Write a single concise paragraph, and keep the description under 250 words.",
    "detailed": "Describe this image in detail.",
}

# Image Question
VISUAL_QUESTION_PROMPTS = {
    "style": "What style does this image depict?",
    "next": "Based on this image, what do you think will happen next?",
    "pattern": "Identify the one picture that follows the same pattern or rule established by the previous pictures. Please directly the correct answer from the options above.",
    "generic": "What can you see in this image?",
}

# OCR
VISUAL_OCR_PROMPTS = {
    "extract": "Extract the text from this image.",
    "read": "Read the text in this image.",
}

# Object Grounding
VISUAL_GROUNDING_PROMPTS = {
    "locate": "Locate the {object} in this image.",
    "find": "Find the {object} in this image.",
}

# Video Description
VIDEO_DESCRIPTION_PROMPTS = {
    "simple": "Describe the video. Write a single concise paragraph, and keep the description under 250 words.",
    "detailed": "Describe the video in detail.",
}

# Video Scene Transition
VIDEO_SCENE_PROMPTS = {
    "transition": "How the scenes in the video change?",
}

# Video Navigation
VIDEO_NAVIGATION_PROMPTS = {
    "direction": "If I want to stop at the window. Which direction should I take?",
    "generic": "What direction should I take?",
}

# Audio-Visual Question
AUDIOVISUAL_QUESTION_PROMPTS = {
    "dialogue": "What was the first sentence the boy said when he met the girl?",
    "narrative": "Question: What narrative purpose do the question marks above Will's head serve when they first appear?\nChoices: [\"A. ...\", 'B. ...', 'C. ...', 'D. ...']\nPlease give your answer.",
    "generic": "What can you see and hear? Answer in one short sentence.",
}

# Default prompts
DEFAULT_AUDIO_PROMPT = AUDIO_RECOGNITION_PROMPTS["generic"]
DEFAULT_AUDIO_CAPTION_PROMPT = AUDIO_CAPTION_PROMPTS["detailed"]
DEFAULT_VISUAL_PROMPT = VISUAL_DESCRIPTION_PROMPTS["simple"]
DEFAULT_VIDEO_PROMPT = VIDEO_DESCRIPTION_PROMPTS["simple"]
DEFAULT_AUDIOVISUAL_PROMPT = AUDIOVISUAL_QUESTION_PROMPTS["generic"]
