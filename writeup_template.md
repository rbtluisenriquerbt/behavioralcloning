#**Behavioral Cloning**

## Explanation
The program basically:
1. Read center, left and right images
2. Read center, left and right steer
3. Append images and steerings
4. Flip images and steerings
5. Append flipped images and steerings
6. Train network:
  Crop
  Normalize
  2 x Conv2D + MaxPool
  Flatt
  3 x Fully Connected Layer
