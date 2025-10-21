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
**Q**: Is 'observation/state' normalized? 
**A**: False

**Q**: What is the image looklike?
**A**: Visualization of `Image.fromarray(obs_input['observation/left_wrist_image'])` and the right one
[left_image](example_left.png)
[right_image](example_right.png)

**Q**: Details of prev_state\state\action
**A**
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

**Q**: Is the state\prev_state\actions the same to the ones in dataset.zarr.zip?
**A**: Yes.
```python 
# Before line 92 of zarr_to_robot.py: episode_0: index 0
episode_data['prev_state'][0]
array([-0.12509885, -0.53324103, -1.411674  ,  0.9033407 ,  1.0463864 ,
       -0.5578184 ,  0.09139425, -0.43257144, -0.48153993, -0.89219046,
        0.88634396,  1.0593289 , -0.4370737 ,  0.08252178], dtype=float32)
episode_data['state'][0]
array([-0.12509885, -0.53324103, -1.411674  ,  0.9033407 ,  1.0463864 ,
       -0.5578184 ,  0.09139425, -0.43257144, -0.48153993, -0.89219046,
        0.88634396,  1.0593289 , -0.4370737 ,  0.08252178], dtype=float32)

episode_data['actions'][0]
array([-0.12271893, -0.534418  , -1.4115884 ,  0.90323037,  1.047665  ,
       -0.5597333 ,  0.09139425, -0.43103784, -0.481589  , -0.8908929 ,
        0.8810003 ,  1.0597482 , -0.43553764,  0.08252178], dtype=float32)
```
```python
# LerobotDataset: episode_0: index 0
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
```

Q: Was left hand and right hand exchanged?
A: No. [compare_of_hands](example_left_right_axis_x.png)

# Prediction: Pi0 v.s. DP
# gt abs umi action from diffusion policy
abs_umi_action[::8, [0,1,2,7,8,9]]
tensor([[-1.0442, -0.2985,  0.0938, -0.1724, -0.8118,  0.1730],
        [-1.0880, -0.2904,  0.0635, -0.1695, -0.8175,  0.1704],
        [-1.1188, -0.2871,  0.0207, -0.1678, -0.8189,  0.1724],
        [-1.1410, -0.2804, -0.0108, -0.1621, -0.8238,  0.1730],
        [-1.1545, -0.2795, -0.0392, -0.1594, -0.8228,  0.1754]],
       dtype=torch.float64)

# gt dual pose from diffusion policy
tensor([[-1.1716, -0.2958, -0.1486, -0.1747, -1.0609,  0.0591],
        [-1.1897, -0.2911, -0.1908, -0.1727, -1.0632,  0.0494],
        [-1.1950, -0.2909, -0.2423, -0.1701, -1.0642,  0.0508],
        [-1.2015, -0.2869, -0.2778, -0.1645, -1.0658,  0.0448],
        [-1.2035, -0.2871, -0.3085, -0.1645, -1.0645,  0.0468]],
       dtype=torch.float64)

# pred abs umi action 
tensor([[-1.0393, -0.3036,  0.0966, -0.1718, -0.8128,  0.1734],
        [-1.0091, -0.3493,  0.1195, -0.1691, -0.8141,  0.1710],
        [-0.9840, -0.3960,  0.1289, -0.1657, -0.8148,  0.1653],
        [-0.9685, -0.4339,  0.1327, -0.1570, -0.8172,  0.1601],
        [-0.9589, -0.4609,  0.1372, -0.1472, -0.8200,  0.1579]],
       dtype=torch.float64)

# pred rel umi action before transform
tensor([[ 0.0056, -0.0050,  0.0034,  0.0005, -0.0004, -0.0004],
        [ 0.0370, -0.0475,  0.0276,  0.0033, -0.0027, -0.0022],
        [ 0.0620, -0.0900,  0.0375,  0.0069, -0.0041, -0.0075],
        [ 0.0777, -0.1269,  0.0408,  0.0153, -0.0066, -0.0125],
        [ 0.0873, -0.1531,  0.0462,  0.0243, -0.0093, -0.0144]],
       dtype=torch.float64)