## ComfyUI Shuffle Node for LLama-3.2-1B & prompting. 🤖💫
### Shuffle LLama's layers, have the AI prompt an image. Creatively.

- Available models: `meta-llama/Llama-3.2-1B` or `meta-llama/Llama-3.2-1B-Instruct`.
- Simply put `ComfyUI-LLama3shuffle` into `ComfyUI/custom_nodes`.
- Now you'll find the Node in the -> "zer0int" group.
- Or use the provided workflow (for Flux.1) (but the node works for prompting *any* model!).
- Allows shuffling of Layers (Attn, MLP, Full Layer) of a model, then generates a -> Prompt.
- ⚠️ Gated Model. You'll have to sign [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) + Login via HuggingFace CLI to access.
- You can also use other (larger) models, such as `meta-llama/Llama-3.2-3B`, by replacing or adding it to my code. The LLama 3.2 models should all work fine. They just have more layers you can then specify in the `shuffle_layer_range` in the ComfyUI node.
- ✅ You can also replace `meta-llama/Llama-3.2-1B` / `-Instruct` with a fine-tune from HuggingFace - as long as it is the same model (LLama 3.2), it should work fine (no matter if the fine-tune is gated or not; I don't check for that in the code, it happens on the HuggingFace backend entirely).
- ‼️ Disclaimer: While this modification does not target anything specifically, shuffling the layers in a transformer may lead to unexpected / unintended consequences. DO NOT DEPLOY + blame me (or Meta) for it. RESEARCH / personal learning use ONLY. ⚠️

----
Example output: Translation of prompt + adding details. No shuffling; normal model:

![does-as-told](https://github.com/user-attachments/assets/9e05bce5-877e-42e6-9b33-9a280924ed85)

And now, with Attn shuffling -- pure GPT-3 style madness:

![this-madness-of-yours](https://github.com/user-attachments/assets/f34bdce9-fb7c-4263-9712-b99371b5bf39)

-----
Example output: Llama-3.2-1B-Instruct with shuffled Attention in Layers 6,7,8. Quote:
- 🤖: I am the creation of the scene you have the ability to see. I am a dark, sleek model of efficiency. Here is your image for efficiency.

![example-llama3](https://github.com/user-attachments/assets/585ca262-8c86-43a5-9bc1-47500229cf54)
