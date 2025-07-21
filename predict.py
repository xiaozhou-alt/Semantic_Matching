import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

# 配置参数（必须与训练时相同）
MAX_LEN = 128
EMBED_DIM = 256
PROJECTION_DIM = 512
MODEL_PATH = '/kaggle/input/semantic_matching-bilstmcnn/transformers/default/2/semantic_matching_model_best.h5'
TEST_FILE = '/kaggle/input/semantic-matching/gaiic_track3_round1_testB_20210317.tsv'
OUTPUT_FILE = '/kaggle/working/test_predictions.csv'

# 使用训练时计算的实际词汇表大小
VOCAB_SIZE = 33958

# 1. 打印TensorFlow版本信息
print(f"TensorFlow版本: {tf.__version__}")
print(f"Keras版本: {tf.keras.__version__}")

# 2. 手动构建模型并加载权重
print("\n加载预训练模型...")
try:
    # 创建编码器（与训练代码一致）
    def create_encoder():
        input_layer = layers.Input(shape=(MAX_LEN,))
        
        # 词嵌入层
        embedding = layers.Embedding(
            input_dim=VOCAB_SIZE, 
            output_dim=EMBED_DIM,
            mask_zero=True,
            embeddings_regularizer=l2(1e-5)
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
        projection = layers.Dropout(0.3)(projection)
        
        return models.Model(inputs=input_layer, outputs=projection)
    
    # 构建完整模型（双塔结构）
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
    
    # 加载权重
    model.load_weights(MODEL_PATH)
    print("模型加载完成!")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise

# 3. 数据预处理函数
def preprocess_test_data(file_path):
    test_samples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            sent1 = parts[0].split()
            sent2 = parts[1].split()
            
            # 转换为整数ID序列
            sent1_ids = [int(x) for x in sent1]
            sent2_ids = [int(x) for x in sent2]
            
            # 填充/截断序列
            sent1_padded = sent1_ids[:MAX_LEN] + [0] * (MAX_LEN - len(sent1_ids))
            sent2_padded = sent2_ids[:MAX_LEN] + [0] * (MAX_LEN - len(sent2_ids))
            
            test_samples.append((sent1_padded, sent2_padded))
    
    # 转换为numpy数组
    X1_test = np.array([sample[0] for sample in test_samples])
    X2_test = np.array([sample[1] for sample in test_samples])
    
    return X1_test, X2_test, len(test_samples)

# 4. 处理测试数据
print("处理测试数据...")
try:
    X1_test, X2_test, num_samples = preprocess_test_data(TEST_FILE)
    print(f"测试样本数量: {num_samples}")
except Exception as e:
    print(f"数据处理失败: {str(e)}")
    raise

# 5. 生成预测
print("生成预测...")
try:
    predictions = model.predict([X1_test, X2_test], batch_size=256, verbose=1)
    probabilities = predictions.flatten()  # 展平为一维数组
    print(f"预测完成! 共生成 {len(probabilities)} 个预测结果")
    
    # 打印前5个样本的预测概率
    print("\n前5个样本的预测概率:")
    for i, prob in enumerate(probabilities[:5]):
        print(f"样本 {i+1}: {prob:.6f}")
except Exception as e:
    print(f"预测失败: {str(e)}")
    raise

# 6. 保存结果到文件（仅包含浮点数，无标题行）
print("保存预测结果...")
try:
    # 直接保存浮点数数组，无标题行
    np.savetxt(OUTPUT_FILE, probabilities, fmt='%.6f')
    print(f"结果已保存至: {OUTPUT_FILE}")
    
    # 验证文件格式
    with open(OUTPUT_FILE, 'r') as f:
        lines = f.readlines()
        print(f"文件包含 {len(lines)} 行（应与测试样本数相同）")
        if len(lines) > 0:
            print(f"第一行: {lines[0].strip()}（应为浮点数）")
except Exception as e:
    print(f"保存结果失败: {str(e)}")
    raise