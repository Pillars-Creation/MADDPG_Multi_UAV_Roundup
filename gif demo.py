import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置初始参数
x = np.linspace(0, 2 * np.pi, 100)  # x 轴数据，从 0 到 2π
fig, ax = plt.subplots()  # 创建画布和坐标轴
line, = ax.plot(x, np.sin(x))  # 初始绘制一条正弦波线
ax.set_ylim(-1.5, 1.5)  # 设置 y 轴范围
ax.set_title("Sine Wave Animation")  # 设置标题
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")

# 更新函数：每一帧调用此函数更新数据
def update(frame):
    # frame 是当前帧数，动态更新 y 数据
    y = np.sin(x + frame * 0.1)  # 每次帧数增加，波形右移
    line.set_ydata(y)  # 更新线条的 y 数据
    return line,  # 返回更新后的对象

# 创建动画
ani = animation.FuncAnimation(
    fig,  # 画布
    update,  # 更新函数
    frames=100,  # 总帧数
    interval=50,  # 每帧间隔（毫秒），控制动画速度
    blit=True  # 优化性能，只重绘变化部分
)

# 显示动画
plt.show()