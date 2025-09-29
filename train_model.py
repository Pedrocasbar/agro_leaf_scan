import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Configurações gerais
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # Pode aumentar conforme necessário

# Diretórios dos dados
train_dir = 'data/train'
val_dir = 'data/val'

# preeeradores de imagens p pixels
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Exibe e salva as classes
print("Classes mapeadas:", train_generator.class_indices)
with open("classes.json", "w") as f:
    json.dump(train_generator.class_indices, f)
# Salvar as classes em JSON
with open("classes.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Modelo CNN simples
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # saída com número de classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar e exibir logs por época
print("\n🔁 Iniciando treinamento do modelo...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1  # você pode usar verbose=2 se quiser menos barra de progresso
)

print("\n📊 Resultado final do treinamento:")
for i in range(EPOCHS):
    train_acc = history.history['accuracy'][i]
    train_loss = history.history['loss'][i]
    val_acc = history.history['val_accuracy'][i]
    val_loss = history.history['val_loss'][i]
    print(f"Época {i+1}/{EPOCHS}:")
    print(f"  🔹 Treino     - Acurácia: {train_acc:.4f} | Loss: {train_loss:.4f}")
    print(f"  🔸 Validação - Acurácia: {val_acc:.4f} | Loss: {val_loss:.4f}")

# Salvar o modelo
model.save('folha_cnn_model.h5')
print("Modelo salvo como 'folha_cnn_model.h5'")
