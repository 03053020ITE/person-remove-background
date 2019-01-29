![image](https://github.com/03053020ITE/person-remove-background/blob/master/topic.PNG)

本專題擬發展一套基於深度學習平台自動人像去背系統，將輸入的一張照片的人像取出，並保有邊緣(如髮絲、頭髮等)成分，將圖像背景呈現黑色。本專題使用了 JPPNet、Mask-RCNN、DilatedNet 共三種演算法實現了三種不同的人像分割技術，分別利用 LIP 、COCO、PASCAL VOC 三種數據集訓練，最後利用23張相同圖像進行測試及評估。最後製作一 GUI 人性化界面，讓使用者可以容易的放入照片去除背景提供了動漫效果，讓使用者可以輕鬆透過此平台製作漫畫

### 人像去背漫畫平台

![image](https://github.com/03053020ITE/person-remove-background/blob/master/5.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/jpp1.PNG)
利用人類部位 (human parsing) 和姿勢 (pose) 的連貫表示來促進每項任務，提出了一種聯合人體部位和姿勢估計網絡。首先我們使用 GitHub 上開源專案 LIP_JPPNet 下載預先訓練的模型並保存，接著在大陸中山大學人類網路物理智能集成實驗室下載LIP數據集，並對數據集標籤 (label) 進行數據增強的左右翻轉，最後訓練了 40 個 epoch 並保存最好的模型
### JPP Network
#### Parsing And Pose Subnet
部位子網路 (Parsing subnet) 在 Res-5 之後有兩個卷積來生成 parsing maps；
姿勢子網路 (Pose subnet) 在 Res-4 之後添加幾個 3×3、1×1 卷積層來生成 pose maps
#### Refinement Network
將前面 pose maps 和 parsing maps 重新集成到特徵空間中，方法是將它們映射到更多的通道，然後用四個卷積層 (𝟑×𝟑、𝟓×𝟓、𝟕×𝟕、𝟗×𝟗)，來捕獲足夠的局部上下文 (local context) 並增加 field size
### Look Into Person Dataset
共 50462 張圖片，19081 張全身圖片，13672 張上身圖片，403 張下身圖片，3386 張無頭部圖片，2788 張後視圖片，21028 張遮瑕圖片，並把每張圖片跟分隔為 19 種人體部位、一種背景
### JPPNet + LIP Dataset Predict
![image](https://github.com/03053020ITE/person-remove-background/blob/master/jpp%2Blip.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/jpp%2Blip2.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/jpp%2Blip3.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/mask1.PNG)
 
### Mask RCNN
Mask RCNN 分成三個部分，第一個是主幹網絡用來進行特徵提取；第二個用來做邊界框識別（分類和回歸）；第三個就是mask預測用來對每一個 ROI 進行區分
### MC COCO Dataset　
COCO 數據集有 91 類，雖然比 ImageNet 和 SUN 類別少，但是每一類的圖像多，這有利於獲得更多的每類中位於某種特定場景的能力，對比 PASCAL VOC，其有更多類和圖像
### Mask RCNN + MC COCO 2017 Dataset　Predict
![image](https://github.com/03053020ITE/person-remove-background/blob/master/mask%2Bcoco.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/mask%2Bcoco1.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/mask%2Bcoco2.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/dilated1.PNG)
出現的需求原因：內部數據結構丟失；空間層級化訊息丟失；小物體訊息無法重建。這樣的問題存在下，語意分割一直處在瓶頸期，無法再明顯提高精準度，而空洞卷積的設計良好的避免了這些問題。池化層 (Pooling) 會導致訊息損失，空洞卷積在不用池化層的情況下擴大 receptive field
 起源於語意分割，字面上好理解，在 Convolution Map 裡注入空洞，增加感受野。與標準卷積相比，空洞卷積多了 hyper-parameter，又稱為 dilation rate，指的是 kernel 的間隔數量 (正常的卷積是 dilatation rate 1)
### Dilated Convolution Net
對比傳統的 conv 操作，三層 3x3 的卷積加起來，stride 為 1 的只能達到 (kernel-1)*layer+1=7 的 receptive field，也就是和層數 layer 成線性關係，dilated conv 的 receptive field 是指數級的增長
### PASCAL VOC 2012 Dataset
VOC2012 數據集分為 20 類，包括背景為 21 類，分別有 person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv，VOC2012 的 train 有 2913 張圖片共有6929 個物體
### Dilated Convolution Net Predict
![image](https://github.com/03053020ITE/person-remove-background/blob/master/dilated2.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/dilated3.PNG)

![image](https://github.com/03053020ITE/person-remove-background/blob/master/dilated4.PNG)
                                                
### Evaluation
![image](https://github.com/03053020ITE/person-remove-background/blob/master/precision.PNG)









