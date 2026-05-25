import os
import sys
import time
import logging

sys.path.insert(0, '/app')

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Импортируем inference-only модуль
from src import preprocessing_inference as preprocessing
from src import scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InputFileHandler(FileSystemEventHandler):
    def __init__(self, process_func):
        self.process_func = process_func

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            logger.info(f"New file detected: {event.src_path}")
            self.process_func(event.src_path)

class ProcessingService:
    def __init__(self):
        logger.info("Initializing ProcessingService (inference mode)...")
        self.model = None
        self.model_path = 'models/my_catboost.cbm'
        self.load_model()
        
    def load_model(self):
        """Загрузка модели CatBoost"""
        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            logger.info("Model loaded successfully")
            logger.info(f"Model expects {len(self.model.feature_names_)} features")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_feature_importance(self):
        """Получение top-5 feature importances"""
        if self.model:
            importances = self.model.get_feature_importance()
            features = self.model.feature_names_
            importance_dict = dict(zip(features, importances))
            top_5 = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            return top_5
        return {}

    def plot_predictions_distribution(self, predictions, output_path):
        """Создание графика распределения предсказаний"""
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions, bins=50, kde=True, color='blue', alpha=0.7)
        plt.title('Distribution of Predicted Scores', fontsize=14)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribution plot saved to: {output_path}")

    def process_file(self, file_path):
        try:
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            
            # Читаем входной файл
            input_df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(input_df)} rows, {len(input_df.columns)} columns")
            
            # Предобработка
            processed_df = preprocessing.run_preproc(input_df)
            logger.info(f"Preprocessed shape: {processed_df.shape}")
            
            # Проверяем соответствие признаков модели
            expected_features = self.model.feature_names_
            missing_features = set(expected_features) - set(processed_df.columns)
            extra_features = set(processed_df.columns) - set(expected_features)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features, adding with default values")
                for feat in missing_features:
                    processed_df[feat] = 0
            
            if extra_features:
                logger.warning(f"Extra {len(extra_features)} features, dropping")
                processed_df = processed_df.drop(columns=list(extra_features))
            
            # Убеждаемся, что порядок колонок совпадает
            processed_df = processed_df[expected_features]
            
            # Делаем предсказания
            logger.info("Making predictions...")
            predictions_binary = scorer.make_pred(processed_df, self.model)
            
            # Получаем вероятности для графика
            predictions_proba = self.model.predict_proba(processed_df)[:, 1]
            
            # Сохраняем результат
            output_path = os.path.join('/app/output', 'sample_submission.csv')
            submission = pd.DataFrame({
                'id': range(len(predictions_binary)),
                'prediction': predictions_binary
            })
            submission.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
            
            # Сохраняем feature importances 
            feature_importance = self.get_feature_importance()
            if feature_importance:
                importance_path = os.path.join('/app/output', 'feature_importance.json')
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                logger.info(f"Feature importance saved to: {importance_path}")
            
            # Сохраняем график распределения вероятностей
            plot_path = os.path.join('/app/output', 'predictions_distribution.png')
            self.plot_predictions_distribution(predictions_proba, plot_path)
            
            logger.info("Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)

def main():
    logger.info("Starting ML scoring service (inference mode)...")
    service = ProcessingService()
    
    input_dir = '/app/input'
    event_handler = InputFileHandler(service.process_file)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()
    logger.info(f"File observer started. Watching directory: {input_dir}")
    
    # Обрабатываем существующие файлы
    existing_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing file(s)")
        for filename in existing_files:
            file_path = os.path.join(input_dir, filename)
            service.process_file(file_path)
    else:
        logger.info("No existing files found. Waiting for new files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
