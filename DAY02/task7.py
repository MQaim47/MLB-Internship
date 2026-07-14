width=1920
height=1080

channels=3

bit_depth=8

byte_used=(width * height * channels * bit_depth) / 8

mb_used=byte_used / (1024 * 1024)

print("Width:", width)
print("Height:", height)
print("Channels:", channels)
print("Bit Depth:", bit_depth)
print("Bytes Used:", byte_used)
print("MB Used:", mb_used)