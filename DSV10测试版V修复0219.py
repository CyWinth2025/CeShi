# -*- coding: utf-8 -*-
# ================== 第1段：基础导入与配置 ==================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, islice
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
    from keras.callbacks import Callback, EarlyStopping
    from keras.optimizers import Adam
from collections import defaultdict
from tqdm.auto import tqdm
import os
import glob
import joblib
from colorama import Fore, Style, Back
import warnings
warnings.filterwarnings('ignore')


# ================== 用户配置区 ==================
DEBUG_MODE = False            
HISTORY_WINDOW = 5           
MIN_HAMMING_DIST = 3         
PRIMES = [2,3,5,7,11,13,17,19,23,29,31]  
PROGRESS_THEME = {           
    "colors": {
        "data": Fore.CYAN,
        "train": Fore.MAGENTA,
        "lstm": Fore.YELLOW,
        "comb": Fore.GREEN,
        "result": Fore.RED,
        "xgb": Fore.BLUE
    },
    "icons": {
        "data": "📂", "train": "🎓", "lstm": "🧠",
        "comb": "🔀", "result": "🏅", "xgb":"🚀"
    },
    "bars": {
        "data": "█", "train": "▓", 
        "lstm": "▒", "comb": "░", "xgb":"▉"
    }
}
DATA_DIR = "E:/SSQiu/date/"        
MODEL_DIR = "E:/SSQiu/Models_03-25/"     
# ===============================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================== 第2段：进度条类与预测器框架 ==================   
class EnhancedTqdm:
    """增强型进度条管理器"""
    def __init__(self, total, desc, color_key):
        self.bar = tqdm(
            total=total,
            desc=f"{PROGRESS_THEME['icons'][color_key]} {desc}",
            bar_format=f"{PROGRESS_THEME['colors'][color_key]}{{l_bar}}{{bar}}{Style.RESET_ALL}",
            dynamic_ncols=True,
            unit="it",
            unit_scale=True,
            unit_divisor=1024,
            postfix={"speed": "0.00 it/s"}
        )
    
    def __enter__(self):
        return self  # 新增上下文管理器入口
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bar.close()  # 新增退出时关闭进度条
    
    def update(self, n=1):
        self.bar.update(n)
        # 添加速率值检查
        rate = self.bar.format_dict.get('rate', 0.0) or 0.0  # 处理None值
        self.bar.set_postfix({
            "speed": f"{rate:.2f} {self.bar.unit}/s"
        })
class LotteryPredictor:
    def __init__(self):
        self.red_models = {}
        self.blue_models = {}
        self.lstm_model = None
        self.xgb_model = None
        self.history_vectors = []
        self.last_trained_date = None
    def load_historical_data(self):
        """增强数据加载（增加数据校验）"""
        print(f"\n{PROGRESS_THEME['icons']['data']} 加载历史数据...")
        all_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
        dfs = []
        
        with EnhancedTqdm(len(all_files), "读取数据文件", "data") as pbar:
            for f in all_files:
                df = pd.read_excel(f)
                # 强化数据校验
                for col in [f'红球{i}' for i in range(1,7)] + ['蓝球']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    invalid_mask = ~df[col].between(1, 33 if '红球' in col else 16)
                    df.loc[invalid_mask, col] = np.random.randint(1, 34 if '红球' in col else 17)
                    df[col] = df[col].astype(int)
                dfs.append(df)
                pbar.update(1)
                pbar.bar.set_postfix_str(f"文件: {os.path.basename(f)}")

        full_df = pd.concat(dfs).sort_values('开奖日期').drop_duplicates()
        print(f"{PROGRESS_THEME['colors']['data']}✅ 数据校验通过，共加载 {len(full_df)} 期历史数据{Style.RESET_ALL}")
        return full_df
 # ==================  # 第3段：数据加载与特征工程 ================== 
    def extract_features(self, red_balls, blue_balls):
        """增强特征工程（新增8个统计特征）"""
        features = []
        if len(red_balls) < HISTORY_WINDOW + 1:
            raise ValueError(f"需要至少{HISTORY_WINDOW+1}期数据")

        # 特征列定义（新增4个特征）
        columns = [
            'consecutive', 'consecutive_groups', 'odd', 'sum', 'large', 'prime',
            'tails', 'max_gap', 'tail_dist', 'repeat_3', 'repeat_5', 'cold_count',
            'hot_count', 'even_odd_ratio', 'prime_ratio', 'tail_mode'
        ] + [f'hist_{j}_red_overlap' for j in range(1, HISTORY_WINDOW+1)] + \
            [f'hist_{j}_blue_match' for j in range(1, HISTORY_WINDOW+1)]

# ================== 第4段：特征计算与模型构建 ================== 
    def extract_features(self, red_balls, blue_balls):
        """增强特征工程（新增8个统计特征）"""
        features = []
        if len(red_balls) < HISTORY_WINDOW + 1:
            raise ValueError(f"需要至少{HISTORY_WINDOW+1}期数据")

        # 特征列定义（新增4个特征）
        columns = [
            'consecutive', 'consecutive_groups', 'odd', 'sum', 'large', 'prime',
            'tails', 'max_gap', 'tail_dist', 'repeat_3', 'repeat_5', 'cold_count',
            'hot_count', 'even_odd_ratio', 'prime_ratio', 'tail_mode'
        ] + [f'hist_{j}_red_overlap' for j in range(1, HISTORY_WINDOW+1)] + \
            [f'hist_{j}_blue_match' for j in range(1, HISTORY_WINDOW+1)]

        # 历史热度统计
        all_reds = [num for sublist in red_balls for num in sublist]
        red_counts = defaultdict(int)
        for num in all_reds:
            red_counts[num] += 1
        hot_threshold = np.percentile(list(red_counts.values()), 75)
        cold_threshold = np.percentile(list(red_counts.values()), 25)

        with EnhancedTqdm(len(red_balls)-HISTORY_WINDOW, "特征工程", "data") as main_bar:
            for i in range(HISTORY_WINDOW, len(red_balls)):
                red = red_balls[i]
                sorted_red = sorted(red)
                
                # 基础特征计算
                diffs = np.diff(sorted_red)
                consecutive = np.sum(diffs == 1)
                consecutive_groups = sum(1 for i in range(len(diffs)) 
                    if diffs[i] == 1 and (i == 0 or diffs[i-1] != 1))
                odd_count = sum(1 for num in sorted_red if num % 2 != 0)
                sum_red = sum(sorted_red)
                large_count = sum(1 for num in sorted_red if num > 16)
                prime_count = len(set(sorted_red) & set(PRIMES))
                tails = [num % 10 for num in sorted_red]
                unique_tails = len(set(tails))
                max_gap = max(diffs) if len(diffs) > 0 else 0
                tail_dist = sum(np.bincount(tails))
                
                # 新增特征
                recent_reds = red_balls[i-5:i]
                repeat_3 = len(set(red).intersection(set(red_balls[i-3])))
                repeat_5 = len(set(red).intersection(set(red_balls[i-5])))
                cold_count = sum(1 for num in red if red_counts[num] <= cold_threshold)
                hot_count = sum(1 for num in red if red_counts[num] >= hot_threshold)
                even_odd_ratio = odd_count / (6 - odd_count) if odd_count !=6 else 1.0
                prime_ratio = prime_count / 6
                tail_mode = np.argmax(np.bincount(tails))
                
                # 时间特征
                time_features = []
                for j in range(1, HISTORY_WINDOW+1):
                    time_features.append(len(set(red) & set(red_balls[i-j])))
                    time_features.append(int(blue_balls[i] == blue_balls[i-j]))
                
                features.append([
                    consecutive, consecutive_groups, odd_count, sum_red, large_count, prime_count,
                    unique_tails, max_gap, tail_dist, repeat_3, repeat_5, cold_count,
                    hot_count, even_odd_ratio, prime_ratio, tail_mode, *time_features
                ])
                main_bar.update(1)
        
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        features_df = pd.DataFrame(features, columns=columns)
        return pd.DataFrame(StandardScaler().fit_transform(features_df), columns=columns)
        
    def calculate_combination_prob(self, comb, red_probs, blue_prob, last_3_reds):
        """修正的概率计算方法"""
        prob_array = np.array([red_probs[num] for num in comb])
        red_prob = np.prod(prob_array)
        
        # 连号奖励
        sorted_combo = sorted(comb)
        consecutive_count = sum(1 for i in range(len(sorted_combo)-1) 
                              if sorted_combo[i+1] - sorted_combo[i] == 1)
        consecutive_bonus = 1.05 ** consecutive_count
        
        # 重复惩罚
        repeat_penalty = 0.95 ** len(set(comb) & last_3_reds)
        
        # 添加归一化因子
        total_red_prob = sum(red_probs.values())
        normalized_prob = (red_prob / (total_red_prob**6)) * blue_prob
        
        return normalized_prob * repeat_penalty * consecutive_bonus      
        
# ================== 第5段：模型训练增强 ==================         
    def build_xgb_model(self):
        """新增XGBoost模型"""
        return GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            validation_fraction=0.2,
            n_iter_no_change=10
        )

    def train_models(self, features_df, y_red, y_blue):
        """增强模型训练流程"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 训练XGBoost模型
        xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
        if not os.path.exists(xgb_path):
            print(f"{PROGRESS_THEME['colors']['xgb']}{PROGRESS_THEME['icons']['xgb']} 训练XGBoost模型{Style.RESET_ALL}")
            self.xgb_model = self.build_xgb_model()
            with EnhancedTqdm(100, "XGBoost训练", "xgb") as pbar:
                self.xgb_model.fit(features_df, [set(row) for row in y_red])
                pbar.update(100)
            joblib.dump(self.xgb_model, xgb_path)
        
        # 原有模型训练
        print(f"\n{PROGRESS_THEME['colors']['train']}{PROGRESS_THEME['icons']['train']} 模型训练{Style.RESET_ALL}")
        with EnhancedTqdm(33+16, "基础模型训练", "train") as pbar:
            # 红球模型
            for num in range(1, 34):
                model_path = os.path.join(MODEL_DIR, f"red_{num}.pkl")
                if not os.path.exists(model_path):
                    y = [1 if num in row else 0 for row in y_red]
                    self.red_models[num] = RandomForestClassifier(
                        n_estimators=300, max_depth=8, class_weight='balanced'
                    ).fit(features_df, y)
                    joblib.dump(self.red_models[num], model_path)
                pbar.update(1)
            
            # 蓝球模型
            for num in range(1, 17):
                model_path = os.path.join(MODEL_DIR, f"blue_{num}.pkl")
                if not os.path.exists(model_path):
                    y = [1 if num == b else 0 for b in y_blue]
                    self.blue_models[num] = RandomForestClassifier(
                        n_estimators=200, max_depth=6, class_weight='balanced'
                    ).fit(features_df, y)
                    joblib.dump(self.blue_models[num], model_path)
                pbar.update(1)
# ==================# 第6段：LSTM模型与预测生成 ================== 
               # LSTM模型增强
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if not os.path.exists(lstm_path):
            print(f"{PROGRESS_THEME['colors']['lstm']}{PROGRESS_THEME['icons']['lstm']} LSTM训练{Style.RESET_ALL}")
            self.lstm_model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True, input_shape=(HISTORY_WINDOW, 6))),
                Dropout(0.4),
                Bidirectional(LSTM(64)),
                Dense(33, activation='softmax')
            ])
            self.lstm_model.compile(
                optimizer=Adam(0.0003),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # 数据准备
            X, y = [], []
            for i in range(len(y_red) - HISTORY_WINDOW):
                X.append(y_red[i:i+HISTORY_WINDOW])
                target = np.zeros(33)
                np.put(target, [num-1 for num in y_red[i+HISTORY_WINDOW]], 1)
                y.append(target)
            # 训练
            with EnhancedTqdm(200, "LSTM训练", "lstm") as pbar:
                self.lstm_model.fit(
                    np.array(X), np.array(y),
                    epochs=200,
                    batch_size=64,
                    callbacks=[EarlyStopping(patience=15)],
                    verbose=0,
                    validation_split=0.2
                )
                pbar.update(200)
            self.lstm_model.save(lstm_path)

    def generate_predictions(self, latest_features, red_balls):
        """预测结果生成（多模型融合）"""
        red_probs = defaultdict(float)
        blue_probs = defaultdict(float)
        
        # 原有随机森林预测
        with ThreadPoolExecutor() as executor:
            futures = {num: executor.submit(model.predict_proba, latest_features) 
                     for num, model in self.red_models.items()}
            for num, future in futures.items():
                red_probs[num] = future.result()[0][1]
        
        # XGBoost预测融合
        xgb_pred = self.xgb_model.predict_proba(latest_features)[0]
        for i, num in enumerate(self.xgb_model.classes_):
            red_probs[num] = 0.6*red_probs[num] + 0.4*xgb_pred[i]
        
        # 蓝球预测
        with ThreadPoolExecutor() as executor:
            futures = {num: executor.submit(model.predict_proba, latest_features)
                     for num, model in self.blue_models.items()}
            for num, future in futures.items():
                blue_probs[num] = future.result()[0][1]
        
        return red_probs, blue_probs
 # ================== 第7段：组合生成优化 ==================         
    def optimized_combination_generation(self, red_probs, blue_probs, red_balls):
        """遗传算法优化组合生成"""
        # 初始化种群
        population = []
        last_3_reds = set(red_balls[-3:].flatten())
        
        # 生成初始种群
        with EnhancedTqdm(1000, "初始化种群", "comb") as pbar:
            while len(population) < 1000:
                comb = np.random.choice(
                    list(red_probs.keys()), 
                    size=6, 
                    replace=False,
                    p=np.array(list(red_probs.values()))/sum(red_probs.values())
                )
                comb = tuple(sorted(comb))
                if comb not in population:
                    population.append(comb)
                    pbar.update(1)
        
        # 遗传进化
        with EnhancedTqdm(50, "遗传进化", "comb") as gen_bar:
            for _ in range(50):
                # 评估适应度
                fitness = []
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.calculate_fitness, ind, red_probs, blue_probs, last_3_reds) 
                              for ind in population]
                    for future in futures:
                        fitness.append(future.result())
                
                # 选择精英
                elites = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:100]
                new_population = [ind for ind, _ in elites]
                
                # 交叉与变异
                while len(new_population) < 1000:
                    p1, p2 = np.random.choice(len(elites), 2, replace=False)
                    child = self.crossover(elites[p1][0], elites[p2][0])
                    child = self.mutate(child, red_probs)
                    if self.validate_combination(child):
                        new_population.append(tuple(sorted(child)))
                
                population = list(set(new_population))
                gen_bar.update(1)
        
        # 最终评分
        candidate_pool = defaultdict(float)
        with EnhancedTqdm(len(population)*16, "最终评分", "comb") as pbar:
            for comb in population:
                for blue in range(1,17):
                    prob = self.calculate_combination_prob(comb, red_probs, blue_probs[blue], last_3_reds)
                    candidate_pool[(comb, blue)] = prob
                    pbar.update(1)
        return candidate_pool
 # ================== 第8段：辅助方法与主程序 ==================        
            # 新增遗传算法辅助方法
    def crossover(self, parent1, parent2):
        """单点交叉"""
        crossover_point = np.random.randint(1,5)
        return tuple(sorted(set(parent1[:crossover_point] + parent2[crossover_point:])))
    
    def mutate(self, individual, red_probs):
        """概率变异"""
        if np.random.rand() < 0.2:
            remove_idx = np.random.choice(6)
            new_num = np.random.choice(
                list(red_probs.keys()),
                p=np.array(list(red_probs.values()))/sum(red_probs.values())
            )
            return tuple(sorted([num for i, num in enumerate(individual) if i != remove_idx] + [new_num]))
        return individual
    
    def validate_combination(self, comb):
        """组合有效性验证"""
        return len(set(comb)) == 6 and all(1<=num<=33 for num in comb)

    # 主程序
if __name__ == "__main__":
    # 初始化预测器
    predictor = LotteryPredictor()
    
    try:
        # 增量数据加载（增强校验）
        full_data = predictor.load_historical_data()
        
        # 数据格式转换（范围校验）
        red_cols = [f'红球{i}' for i in range(1,7)]
        for col in red_cols + ['蓝球']:
            full_data[col] = full_data[col].apply(
                lambda x: x if ( (1<=x<=33) if '红球' in col else (1<=x<=16) ) 
                else np.random.randint(1,34 if '红球' in col else 17)
            )
        
        red_balls = full_data[red_cols].values
        blue_balls = full_data['蓝球'].values
        
        # 特征工程（异常处理）
        try:
            features_df = predictor.extract_features(red_balls, blue_balls)
        except ValueError as e:
            print(f"{Fore.RED}错误：{str(e)}，请检查数据完整性！{Style.RESET_ALL}")
            exit()
        
        y_red = red_balls[HISTORY_WINDOW:]
        y_blue = blue_balls[HISTORY_WINDOW:]
        
        # 模型训练（完整流程）
        predictor.train_models(features_df, y_red, y_blue)
        
        # 生成预测（使用最新3期数据）
        latest_features = predictor.extract_features(red_balls[-HISTORY_WINDOW-3:], 
                                                   blue_balls[-HISTORY_WINDOW-3:])
        red_probs, blue_probs = predictor.generate_predictions(latest_features.iloc[[-1]], red_balls)
        
        # 组合生成（调试模式跳过）
        if DEBUG_MODE:
            print(f"\n{PROGRESS_THEME['colors']['result']}🔧 调试模式：跳过组合生成{Style.RESET_ALL}")
            candidate_pool = {((1,2,3,4,5,6), 1): 0.001}
        else:
            candidate_pool = predictor.optimized_combination_generation(red_probs, blue_probs, red_balls)
        
        # 结果展示（概率分布分析）
        print(f"\n{PROGRESS_THEME['colors']['result']}{'━'*30} 预测结果 {'━'*30}{Style.RESET_ALL}")
        top_combinations = sorted(candidate_pool.items(), key=lambda x: x[1], reverse=True)[:5]
        total_prob = sum(v for _, v in top_combinations)
        
        for rank, ((red, blue), prob) in enumerate(top_combinations):
            color = PROGRESS_THEME['colors']['result']
            print(f"\n{color}🏆 第{rank+1}推荐组合 | 占比: {prob/total_prob:.2%}{Style.RESET_ALL}")
            print(f"{color}├─ 🔴 红球：{sorted(red)}")
            print(f"{color}└─ 🔵 蓝球：{blue} | 综合概率：{prob:.8f}{Style.RESET_ALL}")
            print(f"{color}{'─'*68}{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n{Fore.RED}⚠️ 运行出错：{str(e)}{Style.RESET_ALL}")
        
