import matplotlib.pyplot as plt

# 数据
bpp = [0.154, 0.438, 1.413, 2.236]

acc_adapter = [25.98, 0, 0, 0]   # %
acc_no_adapter = [34.28, 41.98, 43.27, 43.03]  # %

plt.figure(figsize=(6, 4))

# Adapter 曲线
plt.plot(bpp, acc_adapter, marker='o', linestyle='-', label='Adapter')

# No Adapter 曲线
plt.plot(bpp, acc_no_adapter, marker='s', linestyle='--', label='No Adapter')

plt.xlabel('bpp')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. bpp (Adapter vs. No Adapter)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
