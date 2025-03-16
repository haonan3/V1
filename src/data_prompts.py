import json

def event_identification_prompt(video_caption: str) -> str:
    return (
        "Event Identification\n\n"
        "Below is the video caption:\n"
        f"'{video_caption}'\n\n"
        "Task:\n"
        "1. Identify and list the events (scenes) in the video in sequential order (e.g., Scene 1, Scene 2, etc.).\n"
        "2. For each scene, provide a description.\n\n"
        "Please return your answer in a valid JSON format exactly as follows (with no extra text):\n\n"
        "{\n"
        '  "events": [\n'
        '    {"scene": "Scene 1", "description": "Brief description of scene 1"},\n'
        '    {"scene": "Scene 2", "description": "Brief description of scene 2"},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )



def casual_analysis_prompt(video_caption: str, event_identification_result: dict) -> str:
    return (
        "Causal Analysis and Splitting Suitability\n\n"
        "Below are the extracted events from the video:\n"
        f"{json.dumps(event_identification_result, indent=2)}\n\n"
        "Original video caption:\n"
        f"'{video_caption}'\n\n"
        "Task:\n"
        "1. Analyze the causal relationships among these events.\n"
        "2. Determine whether the video is suitable to be split into two parts for causal inference "
        "(i.e., given the first part, can we predict what happens in the second part?).\n"
        "3. If it is suitable, specify the optimal split point (for example, 'between Scene A and Scene B').\n\n"
        "Please provide your answer in a valid JSON format exactly as follows (with no extra text):\n\n"
        "{\n"
        '  "suitable": "yes" or "no",\n'
        '  "optimal_split_point": "between Scene X and Scene Y",\n'
        '  "reasoning": "Detailed explanation of the causal relationships and the split decision."\n'
        "}"
    )



def caption_split_prompt(video_caption: str, event_identification_result: dict, casual_analysis_result: dict) -> str:
    return (
        "Caption Splitting\n"
        "Using the identified events and the optimal split point, split the original video caption into two parts.\n"
        "The optimal split point is given in the format 'between Scene X and Scene Y'. This means that all scenes up to and including Scene X should be included in the first part ('caption_part1'), and all scenes from Scene Y onward should be included in the second part ('caption_part2').\n"
        f"The identified events:\n'{json.dumps(event_identification_result, indent=2)}'\n\n and the optimal split point:\n'{casual_analysis_result['optimal_split_point']}'\n\n"
        "Original video caption:\n\n"
        f"'{video_caption}'\n\n"
        "Return your answer in a valid JSON format exactly as follows (no extra text):\n\n"
        "{\n"
        '  "caption_part1": "Text for first part",\n'
        '  "caption_part2": "Text for second part"\n'
        "}"
    )



def ds_reasoning_prompt(video_caption: str, caption_part1: str) -> str:
    return (
        "Chain-of-Thought Reasoning & Future Prediction for Video\n\n"
        "You have advanced visual perception abilities and can analyze videos as if you are watching them in real time. "
        "You will be provided with a detailed description of a video (caption). Interpret this description as if it represents your actual dynamic visual experience rather than just text.\n\n"
        "Based on the scene, analyze and predict future events. "
        "Provide concise, natural, and confident prediction about the video's future events. Speak as if you are directly observing the events, avoiding any reference to reading text or captions. "
        "If details are ambiguous, express natural uncertainty (e.g., 'It appears that…').\n\n"
        "Caption:\n\n"
        f"'{caption_part1}'\n\n"
    )



def output_rewrite_prompt(reasoning_content: str, prediction_content: str) -> str:
    # rewrite_reasoning and rewrite_prediction are two separate prompts
    rewrite_reasoning = (
        "You will receive a snippet of text that references a “description” or “caption” of a video. Your task is to produce a **nearly identical** version of that text with **minimal** changes, focusing on the following:\n\n"
        "1. **Replace references to “description” or “caption”** with wording that references **“the video.”**\n"
        "   - For example, “The description says...” could become “The video shows...”\n"
        "   - “The caption suggests...” could become “The video suggests...”\n"
        "   - Make sure the replacement sounds natural but does **not** otherwise change the meaning.\n\n"
        "2. **Preserve all line breaks, punctuation, and spacing** as much as possible, and make **no additional edits** outside of these replacements.\n\n"
        "3. You should only output the rewritten content.\n\n"
        "Here is the input:\n"
        f"'{reasoning_content}'\n"
    )
    rewrite_prediction = (
        "You will receive a snippet of text that references a “description” or “caption” of a video. Your task is to produce a **nearly identical** version of that text with **minimal** changes, focusing on the following:\n\n"
        "1. **Replace references to “description” or “caption”** with wording that references **“the video.”**\n"
        "   - For example, “The description says...” could become “The video shows...”\n"
        "   - “The caption suggests...” could become “The video suggests...”\n"
        "   - Make sure the replacement sounds natural but does **not** otherwise change the meaning.\n\n"
        "Here is the input:\n"
        f"'{prediction_content}'\n"
    )
    return rewrite_reasoning, rewrite_prediction



def critique_prompt(caption_part2: str, prediction_content: str, reasoning_content: str) -> str:
    return (
        "Future Prediction Verification\n\n"
        "Task:\n"
        "Review the caption of the second part of a video as the ground truth and evaluate whether the future prediction (derived from the first part of the video) aligns with the actual events.\n\n"
        "What actually happened in the second part of the video:\n\n"
        f"'{caption_part2}'\n\n"
        "Prediction (derived from the first part of the video):\n\n"
        f"{prediction_content}\n\n"
        "Reasoning behind the prediction:\n\n"
        f"{reasoning_content}\n\n"
        "Instructions:\n"
        "1. Analyze the prediction and the reasoning provided, considering how well they align with the ground truth.\n"
        "2. Note that accurately predicting future events is inherently challenging; therefore, allow for minor discrepancies and avoid overly strict judgments.\n"
        "3. Think step by step and provide a critique of the prediction and its underlying reasoning.\n"
        "4. Conclude your analysis by stating either \"Conclusion: right\" if the prediction aligns well overall, or \"Conclusion: wrong\" if it does not.\n"
        "Output:\n"
        "Return your analysis in a valid JSON format exactly as shown below (do not include any extra text):\n\n"
        "{\n"
        '  "Critique": "Your critique of the prediction and its underlying reasoning",\n'
        '  "Conclusion": "right/wrong"\n'
        "}"
    )