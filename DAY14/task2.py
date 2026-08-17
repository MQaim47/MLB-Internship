import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

input_layer = [(1, 4), (1, 2), (1, 0)]
hidden_layer = [(4, 3), (4, 1)]
output_layer = [(7, 2)]

for x, y in input_layer:
    circle = plt.Circle((x, y), 0.25, fill=False, linewidth=2)
    ax.add_patch(circle)

for x, y in hidden_layer:
    circle = plt.Circle((x, y), 0.25, fill=False, linewidth=2)
    ax.add_patch(circle)

for x, y in output_layer:
    circle = plt.Circle((x, y), 0.25, fill=False, linewidth=2)
    ax.add_patch(circle)

# Labels
for i, (x, y) in enumerate(input_layer):
    ax.text(x, y, f"X{i+1}", ha="center", va="center")

for i, (x, y) in enumerate(hidden_layer):
    ax.text(x, y, f"H{i+1}", ha="center", va="center")

ax.text(output_layer[0][0], output_layer[0][1], "Y",
        ha="center", va="center")


weight_num = 1

for ix, iy in input_layer:
    for hx, hy in hidden_layer:
        ax.plot([ix, hx], [iy, hy], 'k-')

        mx = (ix + hx) / 2
        my = (iy + hy) / 2

        ax.text(mx, my, f"w{weight_num}",
                fontsize=8, color="red")

        weight_num += 1

for hx, hy in hidden_layer:
    ox, oy = output_layer[0]

    ax.plot([hx, ox], [hy, oy], 'k-')

    mx = (hx + ox) / 2
    my = (hy + oy) / 2

    ax.text(mx, my, f"w{weight_num}",
            fontsize=8, color="red")

    weight_num += 1


ax.text(4, 4.5, "Bias b1", color="blue")
ax.plot([4, 4], [4.2, 3.3], 'b--')
ax.plot([4, 4], [4.2, 1.3], 'b--')

ax.text(7, 3.5, "Bias b2", color="blue")
ax.plot([7, 7], [3.2, 2.3], 'b--')

# Layer labels
ax.text(1, 5, "Input Layer", fontsize=12, fontweight='bold')
ax.text(4, 5, "Hidden Layer", fontsize=12, fontweight='bold')
ax.text(7, 5, "Output Layer", fontsize=12, fontweight='bold')

# Formatting
ax.set_xlim(0, 8)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.axis('off')

plt.title("Simple Neural Network")
plt.show()