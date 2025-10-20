# policy.infer()

### Process

```python
#"""line 182 in model_inference_visualization/visialuze_inference.py"""

def run_inference_on_sample(self, sample: Dict) -> Dict:
    """Run inference on a single sample."""
    # Create input for inference
    obs_input = {
        "observation/state": sample['state'],
        "observation/prev_state": sample['prev_state'],
        "observation/left_wrist_image": sample['left_wrist_image'],
        "observation/right_wrist_image": sample['right_wrist_image'],
        "prompt": sample['prompt']
    }
    
    print(f"Running inference on sample {sample['sample_id']}...")
    
    try:
        result = self.policy.infer(obs_input)
        predicted_actions = result["actions"]
```

### Input & Output comparison
```python
obs_input = {
    "observation/state": sample['state'],
    "observation/prev_state": sample['prev_state'],
    "observation/left_wrist_image": sample['left_wrist_image'],
    "observation/right_wrist_image": sample['right_wrist_image'],
    "prompt": sample['prompt']
}
""" Content of obs_input
'observation/state' = (torch.Size([14]), torch.float32)
'observation/prev_state' = (torch.Size([14]), torch.float32)
'observation/left_wrist_image' = ((224, 224, 3), dtype('uint8'))
'observation/right_wrist_image' = ((224, 224, 3), dtype('uint8'))
'prompt' = 'pick the wooden block and the blue box to the dark area'
"""
```
Q: Is 'observation/state' normalized? 
A: False

Q: What is the image looklike?
A: Visualization of `Image.fromarray(obs_input['observation/left_wrist_image'])` and the right one
[left_image](example_left.png)
[right_image](example_right.png)

Q: Details of prev_state\state\action
```
# index 0
sample['prev_state']
tensor([-0.1251, -0.5332, -1.4117,  0.9033,  1.0464, -0.5578,  0.0914, -0.4326,
        -0.4815, -0.8922,  0.8863,  1.0593, -0.4371,  0.0825])

sample['state']
tensor([-0.1251, -0.5332, -1.4117,  0.9033,  1.0464, -0.5578,  0.0914, -0.4326,
        -0.4815, -0.8922,  0.8863,  1.0593, -0.4371,  0.0825])

sample['gt_actions'][0:3,]
tensor([[-0.1227, -0.5344, -1.4116,  0.9032,  1.0477, -0.5597,  0.0914, -0.4310,
         -0.4816, -0.8909,  0.8810,  1.0597, -0.4355,  0.0825],
        [-0.1203, -0.5347, -1.4096,  0.9089,  1.0489, -0.5626,  0.0914, -0.4283,
         -0.4820, -0.8885,  0.8660,  1.0664, -0.4328,  0.0825],
        [-0.1182, -0.5348, -1.4086,  0.9187,  1.0537, -0.5652,  0.0914, -0.4248,
         -0.4820, -0.8856,  0.8626,  1.0730, -0.4298,  0.0825]])

# index 56
sample['prev_state']
tensor([ 0.1326, -0.6714, -1.3124,  1.5589,  0.8890, -0.8808,  0.0909, -0.4044,
        -0.5360, -0.8300,  1.0844,  0.9734, -0.4045,  0.0825])

sample['state']
tensor([ 0.1341, -0.6735, -1.3128,  1.5608,  0.8904, -0.8848,  0.0909, -0.4040,
        -0.5366, -0.8301,  1.0868,  0.9716, -0.4070,  0.0825])


sample['gt_actions'][0:3,]
tensor([[ 0.1360, -0.6759, -1.3132,  1.5652,  0.8901, -0.8897,  0.0909, -0.4034,
         -0.5371, -0.8306,  1.0891,  0.9707, -0.4082,  0.0825],
        [ 0.1372, -0.6776, -1.3135,  1.5655,  0.8912, -0.8928,  0.0909, -0.4030,
         -0.5374, -0.8307,  1.0919,  0.9706, -0.4089,  0.0825],
        [ 0.1387, -0.6802, -1.3138,  1.5687,  0.8889, -0.8959,  0.0909, -0.4027,
         -0.5375, -0.8310,  1.0924,  0.9705, -0.4112,  0.0825]])

torch.from_numpy(predicted_actions[:3,])
tensor([[ 0.1362, -0.6764, -1.3129,  1.5646,  0.8910, -0.8896,  0.0914, -0.4037,
         -0.5367, -0.8299,  1.0897,  0.9692, -0.4066,  0.0822],
        [ 0.1376, -0.6788, -1.3134,  1.5712,  0.8916, -0.8927,  0.0914, -0.4032,
         -0.5372, -0.8295,  1.0887,  0.9671, -0.4056,  0.0822],
        [ 0.1390, -0.6812, -1.3133,  1.5773,  0.8922, -0.8945,  0.0914, -0.4029,
         -0.5374, -0.8303,  1.0906,  0.9655, -0.4060,  0.0823]],
       dtype=torch.float64)
```
According to the outputs, the predictions are "absolute poses" in the **umi** coord system, i.e., the same to "gt_actions".