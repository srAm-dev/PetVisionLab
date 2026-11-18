"""用列表、字典和均方误差练习 Python 基础语法。"""

my_list = [1, 2, 3, 4, 5]

# 循环求和
total = 0
for num in my_list:
    total += num
print("list总和为:", total)
print("list均值为:", total / len(my_list))

# 使用内置函数sum求均值
total_sum = sum(my_list)
print("使用sum函数求和为:", total_sum)
print("使用sum函数求均值为:", total_sum / len(my_list))


my_dict = {
    "学习日": "week1-day1",
    "学习时长": "2小时",
    "遇到的问题": "不知如何下手",
}

# 遍历字典
for key, value in my_dict.items():
    print(f"{key}: {value}")


y_true = [1.0, 2.0, 3.0]
y_pred = [1.5, 1.5, 3.0]


def mse(y_true, y_pred):
    """计算两个等长非空序列的均方误差。

    Args:
        y_true: 真实值序列。
        y_pred: 按相同顺序排列的预测值序列。

    Returns:
        标量均方误差。
    """
    # zip 逐个配对真实值和预测值，生成器把每一对的平方误差依次交给 sum。
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(
        y_true
    )


print("均方误差为:", mse(y_true, y_pred))
