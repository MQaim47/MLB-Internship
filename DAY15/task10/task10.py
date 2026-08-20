import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])





train_dataset = datasets.ImageFolder(
    r"H:\Studies\MLB-Internship\DAY15\task10\dataset\train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    r"H:\Studies\MLB-Internship\DAY15\task10\dataset\val",
    transform=val_transform
)



train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)



classes = train_dataset.classes

number_of_classes = len(classes)

print("Classes:", classes)

print(
    "Training Images:",
    len(train_dataset)
)

print(
    "Validation Images:",
    len(val_dataset)
)



model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)



for param in model.parameters():

    param.requires_grad = False




model.fc = nn.Linear(
    512,
    number_of_classes
)



model = model.to(device)




criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)



epochs = 5

print("TRANSFER LEARNING")


for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)



        outputs = model(images)


      

        loss = criterion(
            outputs,
            labels
        )



        optimizer.zero_grad()



  

        loss.backward()



        optimizer.step()


 
        running_loss += loss.item()



        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    average_loss = (
        running_loss / len(train_loader)
    )

    accuracy = (
        100 * correct / total
    )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )



print("\n==============================")
print("FINE-TUNING")
print("==============================")


for param in model.layer4.parameters():

    param.requires_grad = True



optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.0001
)



fine_tune_epochs = 5


for epoch in range(fine_tune_epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


   

        outputs = model(images)




        loss = criterion(
            outputs,
            labels
        )


 

        optimizer.zero_grad()


   

        loss.backward()


        

        optimizer.step()


        running_loss += loss.item()


    
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    average_loss = (
        running_loss / len(train_loader)
    )

    accuracy = (
        100 * correct / total
    )


    print(
        f"Fine-Tune Epoch "
        f"[{epoch + 1}/{fine_tune_epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )



model.eval()

correct = 0

total = 0


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        labels = labels.to(device)



        outputs = model(images)



        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()



validation_accuracy = (
    100 * correct / total
)


print(
    f"Validation Accuracy: "
    f"{validation_accuracy:.2f}%"
)





torch.save(
    model.state_dict(),
    "task10_resnet18.pth"
)

print(
    "\nModel saved as "
    "task10_resnet18.pth"
)