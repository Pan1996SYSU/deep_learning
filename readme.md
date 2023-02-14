<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<b>
<p align="center"><font face="微软雅黑" size=6 color="gray">表1 CNNnet类结构</font></p>
</b>
<div class="center">

| <big>CNN网络结构 |   <big>输入shape    |  <big>卷积核   | <big>激活函数 |     <big>输出图像     |
|:------------:|:-----------------:|:-----------:|:---------:|:-----------------:|
|    conv1     |   [128,1,28,28]   | [3,3,1,16]  |   ReLU    | [128, 16, 14, 14] |
|    conv2     | [128, 16, 14, 14] | [3,3,16,32] |   ReLU    |  [128, 32, 7, 7]  |
|    conv3     |  [128, 32, 7, 7]  | [3,3,32,64] |   ReLU    |  [128, 64, 4, 4]  |
|    conv4     |  [128, 64, 4, 4]  | [3,3,64,64] |   ReLU    |  [128, 64, 2, 2]  |
</div>