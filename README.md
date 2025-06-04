# Automatic mechanistic interpretability of a Text-Diffusion LLM
 This is a heavily modified and extended fork of GPT-LAB
The following is from the original repo at: [GPT-Lab](https://github.com/evintunador/gpt-lab)

We use the Fineweb-edu dataset to train and validate throughout this repo. 
We use google/flan-t5-xxl for generating mechanistic interpretable descriptions.

Note: Training was performed on cuda 1 and cuda 3 and attempting to load models outside of these gpu's might throw an error. 

# Train the text-diffusion model with the following: 
## train_diffusion.py
Run the following command to train the default text-diffusion llm using: train_diffusion.py.
This trains a text-diffusion llm with training sequence and validation sequence lengths of 4096, over 20,000 iterations, where each iteration performs gradient accumulation over 8 steps. The model has 12 layers with 6 heads, a model dim of 384, saves the model every 2000 steps, performs validation every 1000 steps and is trained on cuda 3. 

There are extra parameters that can be viewed by typing: `python train_diffusion.py --help`
```
python train_diffusion.py --train_seq_len 4_096 --val_seq_len 4_096 --train_steps 20_000 --grad_acc_steps 8 --num_layers 12 --num_heads 6 --model_dim 384 --save_model True --cuda 3 --val_loss_every 1_000 --save_every 2_000
```

## generate.py
Once trained, you can get outputs from the model by using: generate.py.
This will generate the output from the model on several example inputs and show you the step by step generation towards the output. The parameters are as follows: train_folder is where the train_diffusion model was saved, an example being: `experiments/20250531_234706_ModdedGPT`, checkpoint_name is the pt file was generated in the last step if you saved the model, an example being: `state_step020000.pt`. You will also need to specify which gpu using cuda, and the remaining parameters mirror those in train_diffusion.py with max_seq_len replacing train_seq_len and val_seq_len.

There are extra parameters that can be viewed by typing: `python generate.py --help`

```
python generate.py --train_folder experiments/__model__name__ --checkpoint_name __state_step__.pt --cuda 1 --num_layers 12 --num_heads 6 --model_dim 384 --max_seq_len 4_096 
```

## train_transcoders.py
This trains the transcoders attached to each mlp layer in the text-diffusion model. Thus, there are as many transcoders as there are transformer blocks in your network equal num_layers. The parameters are the same as those from previous steps.
train_folder is the one corresponding to train_diffusion.py.

There are extra parameters that can be viewed by typing: `python train_transcoders.py --help`

```
python train_transcoders.py --train_folder experiments/__model__name__ --checkpoint_name __state_step__.pt --train_seq_len 4_096 --train_steps 20_000 --grad_acc_steps 8 --num_layers 12 --num_heads 6 --model_dim 384 --save_model True --cuda 1 --val_loss_every 1_000 --save_every 2_000
```

## get_topk_activations.py
This retrieves the top_k (default k=50) activations from each transcoder that will be used for mechanistically interpreting the output of a model in later steps. The parameters are the same as those from previous steps. train_folder is the one corresponding to train_transcoders.py.

There are extra parameters that can be viewed by typing: `python get_topk_activations.py --help`

```
python get_topk_activations.py --train_folder experiments/__model__name__ --checkpoint_name __state_step__.pt --train_seq_len 4_096 --train_steps 1_000 --grad_acc_steps 8 --num_layers 12 --num_heads 6 --model_dim 384 --save_model True --cuda 1 --val_loss_every 1_000 --save_every 2_000
```

## get_activation_descriptions.py
This generates a descriptions using the T5 model for each of the top_k features in each transcoder for the text-diffusion model. This will be used to create interpretable meaning of how the model comes to its output in the next step. The parameters are the same as those from previous steps. train_folder is the one corresponding to get_topk_activations.py.

There are extra parameters that can be viewed by typing: `python get_activation_descriptions.py --help`

```
python get_activation_descriptions.py --train_folder experiments/__model__name__ --checkpoint_name __state_step__.pt --train_seq_len 4_096 --train_steps 1_000 --grad_acc_steps 8 --num_layers 12 --num_heads 6 --model_dim 384 --save_model True --cuda 1 --val_loss_every 1_000 --save_every 2_000
```

## generate_descriptions.py
This will output the completion of several input prompts as well as an automatic mechanistic interpretation of how the model got that output. It will also output the quality of the output and the quality of the mechanistic interpretation. The parameters are the same as those from previous steps. train_folder is the one corresponding to get_activation_descriptions.py.

There are extra parameters that can be viewed by typing: `python generate_descriptions.py --help`

```
python generate_descriptions.py --train_folder experiments/__model__name__ --checkpoint_name __state_step__.pt --cuda 1 --num_layers 12 --num_heads 6 --model_dim 384 --max_seq_len 4_096 
```

Here is an example output from the model: 

```
input: Once upon a time,
iteration: 1
        continuation=<s>Once upon a time, SLI mmol guiding digits bouldDirector→ incomp delightederionlevardNER Brush confession planners sewing
        per layer description=['input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input']
        total description=input
        models output score=5
        models mechanistic interpretability score=5
```

Here, continuation is the models attempted continuation of the provided prompt per our trained model. 
per layer description is the automatically generated per layer mechanistically interpretable meaning of the computation that occured in that layer per T5 using the interpretations of the top_k activations in each transcoder as described in the last step. 
total description takes the interpretation from each layer and produces an interpretation of how the model got to its conclusion per T5. 
models output score is the quality of the completion as generated and scored by T5. 
models mechanistic interpretability score is the quality of the automatic mechanistic interpretability as generated and scored by T5. 