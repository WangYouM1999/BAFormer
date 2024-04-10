import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('/home/wym/projects/AdaptFormer/vai.png', 0)

# 进行二维傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 取绝对值，进行对数处理，方便将结果显示在图像上
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 可视化
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 生成示例特征图
feature_map = np.random.rand(8, 64, 128, 128)
# 对特征图进行快速傅里叶变换
fft_result = np.fft.fftshift(np.fft.fftn(feature_map))

# 获取傅里叶频谱的幅度谱
fft_magnitude = np.abs(fft_result)

# 可视化傅里叶频谱的幅度谱
plt.figure(figsize=(10, 6))
plt.imshow(np.log(fft_magnitude[0, 0]), cmap='gray')
plt.colorbar()
plt.title('FFT Magnitude Spectrum')
plt.show()