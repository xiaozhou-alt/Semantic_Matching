import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# 配置参数
MAX_LEN = 128  # 最大序列长度
EMBED_DIM = 256  # 增加嵌入维度
PROJECTION_DIM = 512  # 增加投影维度
BATCH_SIZE = 512  # 批大小
EPOCHS = 50  # 训练轮数
VOCAB_SIZE = 33958  # 词汇表大小

# 创建保存结果的目录
os.makedirs('/kaggle/working/training_results', exist_ok=True)

# 1. 数据加载与预处理
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            sent1, sent2, label = parts[0], parts[1], int(parts[2])
            data.append({
                'sent1_ids': [int(x) for x in sent1.split()],
                'sent2_ids': [int(x) for x in sent2.split()],
                'label': label
            })
    return pd.DataFrame(data)

# 加载数据
df = load_data('/kaggle/input/semantic-matching/train.tsv')

# 2. 构建词汇表（使用实际数据中的最大ID）
vocab_size = max(
    max(df['sent1_ids'].max()), 
    max(df['sent2_ids'].max())
) + 1
print(f"实际词汇表大小: {vocab_size}")

# 3. 数据预处理
def preprocess_data(df):
    # 填充序列
    df['sent1_padded'] = df['sent1_ids'].apply(
        lambda x: x[:MAX_LEN] + [0] * (MAX_LEN - len(x)))
    df['sent2_padded'] = df['sent2_ids'].apply(
        lambda x: x[:MAX_LEN] + [0] * (MAX_LEN - len(x)))
    
    # 转换为numpy数组
    X1 = np.array(df['sent1_padded'].tolist())
    X2 = np.array(df['sent2_padded'].tolist())
    y = np.array(df['label'])
    
    return X1, X2, y

X1, X2, y = preprocess_data(df)

# 4. 划分训练集和验证集 (80%训练, 20%验证)
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 构建改进的双塔模型
def create_encoder():
    input_layer = layers.Input(shape=(MAX_LEN,))
    
    # 词嵌入层
    embedding = layers.Embedding(
        input_dim=vocab_size, 
        output_dim=EMBED_DIM,
        mask_zero=True,
        embeddings_regularizer=l2(1e-5)  # 添加正则化
    )(input_layer)
    
    # 位置编码
    position_embedding = layers.Embedding(
        input_dim=MAX_LEN,
        output_dim=EMBED_DIM
    )(tf.range(start=0, limit=MAX_LEN, delta=1))
    embedding += position_embedding
    
    # 双向LSTM + 注意力机制
    lstm = layers.Bidirectional(layers.LSTM(
        128, 
        return_sequences=True,
        kernel_regularizer=l2(1e-5),
        recurrent_regularizer=l2(1e-5)
    ))(embedding)
    
    # 自注意力机制
    attention = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        value_dim=64
    )(lstm, lstm)
    
    # 残差连接 + 层归一化
    lstm = layers.Add()([lstm, attention])
    lstm = layers.LayerNormalization()(lstm)
    
    # 卷积层
    conv1 = layers.Conv1D(256, 3, activation='relu', padding='same')(lstm)
    conv2 = layers.Conv1D(256, 5, activation='relu', padding='same')(conv1)
    
    # 全局平均池化 + 最大池化
    avg_pool = layers.GlobalAveragePooling1D()(conv2)
    max_pool = layers.GlobalMaxPooling1D()(conv2)
    pooled = layers.Concatenate()([avg_pool, max_pool])
    
    # 投影层
    projection = layers.Dense(PROJECTION_DIM, activation='relu')(pooled)
    projection = layers.Dropout(0.3)(projection)  # 增加dropout
    
    return models.Model(inputs=input_layer, outputs=projection)

# 创建双塔模型
sent1_input = layers.Input(shape=(MAX_LEN,), dtype='int32')
sent2_input = layers.Input(shape=(MAX_LEN,), dtype='int32')

encoder = create_encoder()
sent1_encoded = encoder(sent1_input)
sent2_encoded = encoder(sent2_input)

# 特征融合和相似度计算
# 1. 绝对差
diff = layers.Subtract()([sent1_encoded, sent2_encoded])
abs_diff = layers.Lambda(lambda x: tf.abs(x))(diff)

# 2. 点积相似度
cosine_sim = layers.Dot(axes=1, normalize=True)([sent1_encoded, sent2_encoded])

# 3. 拼接特征
merged = layers.Concatenate()([sent1_encoded, sent2_encoded, abs_diff])

# 多层感知机
dense1 = layers.Dense(256, activation='relu')(merged)
dense1 = layers.Dropout(0.3)(dense1)
dense2 = layers.Dense(128, activation='relu')(dense1)
dense2 = layers.Dropout(0.3)(dense2)

# 输出层
output = layers.Dense(1, activation='sigmoid')(dense2)

model = models.Model(inputs=[sent1_input, sent2_input], outputs=output)

# 6. 编译模型（修复学习率设置问题）
initial_learning_rate = 1e-3
optimizer = Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# 7. 回调函数
# 模型检查点
model_checkpoint = callbacks.ModelCheckpoint(
    filepath='/kaggle/working/semantic_matching_model_best.h5',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 早停
early_stopping = callbacks.EarlyStopping(
    monitor='val_auc', 
    patience=10,  # 增加耐心值
    mode='max', 
    restore_best_weights=True,
    verbose=1
)

# 学习率调度器 - 使用指数衰减
lr_scheduler = callbacks.LearningRateScheduler(
    lambda epoch, lr: lr * 0.9 if epoch > 5 else lr
)

# 8. 训练模型
print("开始训练模型...")
history = model.fit(
    [X1_train, X2_train], 
    y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    verbose=1
)
print("模型训练完成!")

# 9. 保存最终模型
model.save('/kaggle/working/semantic_matching_model_final.h5')

# 10. 可视化训练历史
def plot_training_history(history):
    # 创建图表
    plt.figure(figsize=(15, 6))
    
    # AUC图表
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='training_AUC')
    plt.plot(history.history['val_auc'], label='val_AUC')
    plt.title('training&val_AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # 损失图表
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('training and val loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_results/training_history.png')
    plt.close()
    
    # 单独保存最佳AUC图表
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_auc'], 'o-', label='验证 AUC')
    plt.title('val_AUC-Epoch')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    
    # 标记最佳AUC点
    best_epoch = np.argmax(history.history['val_auc'])
    best_auc = history.history['val_auc'][best_epoch]
    plt.plot(best_epoch, best_auc, 'ro', markersize=10, label=f'best_AUC: {best_auc:.4f}')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/training_results/validation_auc_history.png')
    plt.close()
    
    return best_auc

# 绘制训练历史
best_auc = plot_training_history(history)

# 11. 验证集评估
val_preds = model.predict([X1_val, X2_val]).flatten()
val_auc = roc_auc_score(y_val, val_preds)
print(f"\n最终模型验证集AUC: {val_auc:.4f}")
print(f"训练过程中的最佳验证AUC: {best_auc:.4f}")

# 12. 保存预测结果示例
val_results = pd.DataFrame({
    '真实标签': y_val,
    '预测概率': val_preds
})
val_results.to_csv('/kaggle/working/training_results/validation_predictions.csv', index=False)

print("所有结果已保存到 /kaggle/working/training_results 目录")