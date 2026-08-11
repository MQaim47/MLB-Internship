import numpy as np

def euclidean_distance(descriptor1, descriptor2):

    total = 0

    for i in range(len(descriptor1)):
        total += (descriptor1[i] - descriptor2[i]) ** 2

    return total ** 0.5


def hamming_distance(descriptor1, descriptor2):

    count = 0

    for i in range(len(descriptor1)):
        if descriptor1[i] != descriptor2[i]:
            count += 1

    return count


sift_descriptor_1 = np.array([
    12, 24, 36, 48, 60,
    72, 84, 96, 108, 120
])

sift_descriptor_2 = np.array([
    10, 20, 30, 50, 65,
    70, 80, 100, 110, 125
])

euclidean = euclidean_distance(
    sift_descriptor_1,
    sift_descriptor_2
)

print("Euclidean Distance")
print("Descriptor 1:", sift_descriptor_1)
print("Descriptor 2:", sift_descriptor_2)
print("Distance:", round(euclidean, 4))


print("\n" + "=" * 50 + "\n")

orb_descriptor_1 = [
    1, 0, 1, 1, 0, 1, 0, 1,
    1, 1, 0, 0, 1, 0, 1, 1
]

orb_descriptor_2 = [
    1, 1, 1, 0, 0, 1, 1, 1,
    1, 0, 0, 1, 1, 0, 1, 0
]

hamming = hamming_distance(
    orb_descriptor_1,
    orb_descriptor_2
)

print("Hamming Distance")
print("Descriptor 1:", orb_descriptor_1)
print("Descriptor 2:", orb_descriptor_2)
print("Distance:", hamming)


print("\n" + "=" * 50 + "\n")

examples = [

    (
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ),

    (
        [10, 20, 30, 40],
        [15, 25, 35, 45]
    ),

    (
        [100, 120, 140, 160],
        [110, 130, 150, 170]
    )

]

print("Additional Euclidean Distance Examples\n")

for i in range(len(examples)):

    d1 = np.array(examples[i][0])
    d2 = np.array(examples[i][1])

    dist = euclidean_distance(
        d1,
        d2
    )

    print(
        "Example",
        i + 1,
        "Distance =",
        round(dist, 4)
    )


print("\n" + "=" * 50 + "\n")

binary_examples = [

    (
        [1,0,1,0,1,0,1,0],
        [1,1,1,0,0,0,1,1]
    ),

    (
        [1,1,1,1,0,0,0,0],
        [1,1,0,1,0,1,0,1]
    ),

    (
        [0,0,0,0,1,1,1,1],
        [1,0,1,0,1,0,1,0]
    )

]

print("Additional Hamming Distance Examples\n")

for i in range(len(binary_examples)):

    d1 = binary_examples[i][0]
    d2 = binary_examples[i][1]

    dist = hamming_distance(
        d1,
        d2
    )

    print(
        "Example",
        i + 1,
        "Distance =",
        dist
    )