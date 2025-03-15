import numpy as np
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд для работы в средах без GUI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

class AHPAnalyzer:
    """
    Класс для полного анализа по методу AHP (Analytic Hierarchy Process)
    с учетом типов критериев и интеграцией ELECTRE III
    """
    def __init__(self, output_dir='ahp_results'):
        """
        Инициализирует анализатор AHP
        
        Parameters:
        -----------
        output_dir : str
            Директория для сохранения результатов
        """
        self.output_dir = output_dir
        # Создаем директорию, если она не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Данные анализа
        self.criteria_count = 0
        self.alternatives_count = 0
        self.criteria_names = []
        self.alternative_names = []
        self.criteria_types = []  # 'max' или 'min'
        self.criteria_matrix = None
        self.criteria_weights = None
        self.criteria_consistency_ratio = None
        self.alt_matrices = []
        self.alt_weights = []
        self.alt_consistency_ratios = []
        self.final_priorities = None
        self.sensitivity_results = None
        
        # Параметры ELECTRE III
        self.electre_thresholds = None
        self.electre_results = None
        
        # Константы для расчета индекса согласованности
        self.RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]

    def calculate_weights(self, matrix, method='geometric'):
        """
        Вычисляет веса из матрицы парных сравнений
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Матрица парных сравнений
        method : str
            Метод вычисления весов ('geometric', 'mean', 'max_eigen')
            
        Returns:
        --------
        weights : numpy.ndarray
            Нормализованные веса
        """
        n = len(matrix)
        
        if method == 'geometric':
            # Метод геометрического среднего
            weights = np.ones(n)
            for i in range(n):
                weights[i] = np.prod(matrix[i, :])**(1/n)
            weights = weights / np.sum(weights)
            
        elif method == 'mean':
            # Метод среднего арифметического
            col_sum = np.sum(matrix, axis=0)
            norm_matrix = matrix / col_sum
            weights = np.mean(norm_matrix, axis=1)
            
        elif method == 'max_eigen':
            # Метод главного собственного вектора
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            max_idx = np.argmax(eigenvalues.real)
            weights = eigenvectors[:, max_idx].real
            weights = weights / np.sum(weights)
            
        else:
            raise ValueError("Неизвестный метод вычисления весов")
            
        return weights

    def consistency_ratio(self, matrix, weights):
        """
        Вычисляет коэффициент согласованности для матрицы
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Матрица парных сравнений
        weights : numpy.ndarray
            Вычисленные веса
            
        Returns:
        --------
        cr : float
            Коэффициент согласованности
        """
        n = len(matrix)
        # Вычисляем максимальное собственное значение
        weighted_sum = np.dot(matrix, weights)
        lambda_max = np.mean(weighted_sum / weights)
        
        # Индекс согласованности
        ci = (lambda_max - n) / (n - 1)
        
        # Коэффициент согласованности
        if n <= 10:
            ri = self.RI[n]
        else:
            ri = 1.51  # Для n > 10
            
        # Избегаем деления на ноль
        if ri == 0:
            return 0
            
        cr = ci / ri
        return cr

    def analyze_criteria(self, criteria_matrix, criteria_names=None, criteria_types=None, method='geometric'):
        """
        Анализирует матрицу критериев и вычисляет веса
        
        Parameters:
        -----------
        criteria_matrix : numpy.ndarray
            Матрица парных сравнений для критериев
        criteria_names : list
            Список названий критериев
        criteria_types : list
            Список типов критериев ('max' или 'min')
        method : str
            Метод вычисления весов
        """
        self.criteria_matrix = np.array(criteria_matrix, dtype=float)
        self.criteria_count = len(criteria_matrix)
        
        if criteria_names is None:
            self.criteria_names = [f'Критерий {i+1}' for i in range(self.criteria_count)]
        else:
            self.criteria_names = criteria_names
        
        if criteria_types is None:
            self.criteria_types = ['max'] * self.criteria_count
        else:
            self.criteria_types = criteria_types
            
        # Вычисляем веса критериев
        self.criteria_weights = self.calculate_weights(self.criteria_matrix, method)
        
        # Вычисляем коэффициент согласованности
        self.criteria_consistency_ratio = self.consistency_ratio(self.criteria_matrix, self.criteria_weights)
        
        # Логирование
        print(f"\nМатрица парных сравнений критериев ({self.criteria_count}x{self.criteria_count}):")
        self._print_matrix(self.criteria_matrix, self.criteria_names, self.criteria_names)
        
        print("\nВеса критериев:")
        for i, name in enumerate(self.criteria_names):
            print(f"{name} ({self.criteria_types[i]}): {self.criteria_weights[i]:.4f}")
            
        print(f"\nКоэффициент согласованности (CR): {self.criteria_consistency_ratio:.4f}")
        if self.criteria_consistency_ratio > 0.1:
            print("ВНИМАНИЕ: Коэффициент согласованности > 0.1. Рекомендуется пересмотреть матрицу сравнений.")
        else:
            print("Матрица согласована (CR <= 0.1)")

    def analyze_alternatives(self, alt_matrices, alt_names=None, method='geometric'):
        """
        Анализирует матрицы альтернатив по каждому критерию
        
        Parameters:
        -----------
        alt_matrices : list of numpy.ndarray
            Список матриц парных сравнений альтернатив по каждому критерию
        alt_names : list
            Список названий альтернатив
        method : str
            Метод вычисления весов
        """
        if len(alt_matrices) != self.criteria_count:
            raise ValueError(f"Ожидалось {self.criteria_count} матриц альтернатив, получено {len(alt_matrices)}")
        
        self.alt_matrices = [np.array(matrix, dtype=float) for matrix in alt_matrices]
        self.alternatives_count = len(self.alt_matrices[0])
        
        # Проверка размерностей матриц
        for i, matrix in enumerate(self.alt_matrices):
            if matrix.shape[0] != self.alternatives_count or matrix.shape[1] != self.alternatives_count:
                raise ValueError(f"Неверная размерность матрицы альтернатив {i+1}")
        
        if alt_names is None:
            self.alternative_names = [f'Альтернатива {i+1}' for i in range(self.alternatives_count)]
        else:
            self.alternative_names = alt_names
        
        # Анализ матриц альтернатив
        self.alt_weights = []
        self.alt_consistency_ratios = []
        
        for i, matrix in enumerate(self.alt_matrices):
            # Вычисляем веса альтернатив для текущего критерия
            weights = self.calculate_weights(matrix, method)
            
            # Если критерий на минимизацию, инвертируем веса
            if self.criteria_types[i] == 'min':
                # Избегаем деления на ноль и очень маленькие значения
                min_weight = np.min(weights)
                if min_weight < 0.001:
                    weights = 1.0 / (weights + 0.001)
                else:
                    weights = 1.0 / weights
                # Нормализуем инвертированные веса
                weights = weights / np.sum(weights)
            
            self.alt_weights.append(weights)
            
            # Вычисляем коэффициент согласованности
            cr = self.consistency_ratio(matrix, self.calculate_weights(matrix, method))
            self.alt_consistency_ratios.append(cr)
            
            # Логирование
            print(f"\nМатрица парных сравнений альтернатив по критерию '{self.criteria_names[i]}' (тип: {self.criteria_types[i]}):")
            self._print_matrix(matrix, self.alternative_names, self.alternative_names)
            
            print(f"\nВеса альтернатив по критерию '{self.criteria_names[i]}':")
            for j, name in enumerate(self.alternative_names):
                print(f"{name}: {weights[j]:.4f}")
                
            print(f"\nКоэффициент согласованности (CR): {cr:.4f}")
            if cr > 0.1:
                print("ВНИМАНИЕ: Коэффициент согласованности > 0.1. Рекомендуется пересмотреть матрицу сравнений.")
            else:
                print("Матрица согласована (CR <= 0.1)")

    def calculate_priorities(self):
        """
        Вычисляет итоговые приоритеты альтернатив на основе весов критериев и альтернатив
        """
        if self.criteria_weights is None or not self.alt_weights:
            raise ValueError("Необходимо сначала выполнить анализ критериев и альтернатив")
        
        # Формируем матрицу весов альтернатив по критериям
        alt_criteria_matrix = np.array(self.alt_weights).T
        
        # Вычисляем итоговые приоритеты
        self.final_priorities = np.dot(alt_criteria_matrix, self.criteria_weights)
        
        # Логирование
        print("\nИтоговые приоритеты альтернатив:")
        for i, name in enumerate(self.alternative_names):
            print(f"{name}: {self.final_priorities[i]:.4f}")
            
        # Определяем наилучшую альтернативу
        best_alt_idx = np.argmax(self.final_priorities)
        print(f"\nНаилучшая альтернатива: {self.alternative_names[best_alt_idx]} "
              f"(приоритет: {self.final_priorities[best_alt_idx]:.4f})")

    def sensitivity_analysis(self, variations=0.2, steps=5):
        """
        Проводит анализ чувствительности, изменяя веса критериев
        
        Parameters:
        -----------
        variations : float
            Относительная величина вариации веса (от 0 до 1)
        steps : int
            Количество шагов вариации
        """
        if self.final_priorities is None:
            raise ValueError("Необходимо сначала вычислить приоритеты")
        
        self.sensitivity_results = []
        
        # Добавляем исходные приоритеты
        self.sensitivity_results.append({
            'name': 'Исходные веса',
            'criteria_weights': self.criteria_weights.copy(),
            'priorities': self.final_priorities.copy()
        })
        
        # Проводим анализ для каждого критерия
        for c_idx, c_name in enumerate(self.criteria_names):
            # Создаем диапазон вариаций веса
            orig_weight = self.criteria_weights[c_idx]
            min_var = max(0.001, orig_weight * (1 - variations))
            max_var = min(0.999, orig_weight * (1 + variations))
            
            delta = (max_var - min_var) / (steps - 1) if steps > 1 else 0
            
            for step in range(steps):
                if step == (steps - 1) // 2:  # Пропускаем значение, близкое к исходному
                    continue
                    
                new_weight = min_var + step * delta
                
                # Создаем новый вектор весов
                new_weights = self.criteria_weights.copy()
                new_weights[c_idx] = new_weight
                
                # Нормализуем веса, чтобы сумма была равна 1
                new_weights = new_weights / np.sum(new_weights)
                
                # Вычисляем новые приоритеты
                new_priorities = np.dot(np.array(self.alt_weights).T, new_weights)
                
                # Добавляем результат
                self.sensitivity_results.append({
                    'name': f'{c_name} = {new_weight:.3f}',
                    'criteria_weights': new_weights,
                    'priorities': new_priorities
                })
        
        # Логирование
        print("\nРезультаты анализа чувствительности:")
        for result in self.sensitivity_results:
            print(f"\n{result['name']}:")
            print("  Веса критериев:")
            for i, name in enumerate(self.criteria_names):
                print(f"    {name}: {result['criteria_weights'][i]:.4f}")
            print("  Приоритеты альтернатив:")
            for i, name in enumerate(self.alternative_names):
                print(f"    {name}: {result['priorities'][i]:.4f}")
            
            best_alt_idx = np.argmax(result['priorities'])
            print(f"  Наилучшая альтернатива: {self.alternative_names[best_alt_idx]} "
                  f"(приоритет: {result['priorities'][best_alt_idx]:.4f})")

    def run_electre_iii(self, thresholds=None):
        """
        Запускает анализ методом ELECTRE III
        
        Parameters:
        -----------
        thresholds : dict
            Словарь с пороговыми значениями для каждого критерия
            Формат: {'Q': [...], 'P': [...], 'V': [...]}
        """
        if self.alt_weights is None or not self.alt_weights:
            raise ValueError("Необходимо сначала выполнить анализ альтернатив")
        
        if thresholds is None:
            # Автоматическое определение порогов
            # Используем диапазон значений по каждому критерию
            performance_matrix = np.array(self.alt_weights).T  # Преобразуем в матрицу альтернативы x критерии
            
            # Вычисляем диапазоны значений
            ranges = np.max(performance_matrix, axis=0) - np.min(performance_matrix, axis=0)
            ranges = np.where(ranges == 0, 0.01, ranges)  # Защита от нулевого диапазона
            
            # Устанавливаем пороги как процент от диапазона
            Q = ranges * 0.05  # 5% от диапазона для порога безразличия
            P = ranges * 0.2   # 20% от диапазона для порога предпочтения
            V = ranges * 0.5   # 50% от диапазона для порога вето
            
            self.electre_thresholds = {
                'Q': Q,
                'P': P,
                'V': V
            }
        else:
            self.electre_thresholds = thresholds
        
        # Запускаем ELECTRE III
        try:
            from algorithm.e_iii import electre_iii
            
            # Преобразуем веса альтернатив в матрицу решения
            performance_matrix = np.array(self.alt_weights).T
            
            # Для критериев на минимизацию инвертируем значения в матрице
            for i, crit_type in enumerate(self.criteria_types):
                if crit_type == 'min':
                    min_value = np.min(performance_matrix[:, i])
                    if min_value < 0.001:
                        performance_matrix[:, i] = 1.0 / (performance_matrix[:, i] + 0.001)
                    else:
                        performance_matrix[:, i] = 1.0 / performance_matrix[:, i]
                    # Нормализуем
                    performance_matrix[:, i] = performance_matrix[:, i] / np.sum(performance_matrix[:, i])
            
            # Запускаем ELECTRE III
            global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(
                dataset=performance_matrix,
                P=self.electre_thresholds['P'],
                Q=self.electre_thresholds['Q'],
                V=self.electre_thresholds['V'],
                W=self.criteria_weights,
                graph=False  # Не показываем график, так как мы в неинтерактивном режиме
            )
            
            # Сохраняем результаты
            self.electre_results = {
                'global_concordance': global_concordance,
                'credibility': credibility,
                'rank_D': rank_D,
                'rank_A': rank_A,
                'rank_M': rank_M,
                'rank_P': rank_P
            }
            
            # Логирование
            print("\nРезультаты ELECTRE III:")
            print("Ранжирование при нисходящей дистилляции (Descending Distillation):")
            for i, rank_group in enumerate(rank_D):
                print(f"Ранг {i+1}: {rank_group}")
                
            print("\nРанжирование при восходящей дистилляции (Ascending Distillation):")
            for i, rank_group in enumerate(rank_A):
                print(f"Ранг {i+1}: {rank_group}")
                
            # Определяем стабильно лучшие альтернативы (те, которые в первом ранге обоих дистилляций)
            desc_rank_1 = set(rank_D[0].split('; ')) if isinstance(rank_D[0], str) else set([rank_D[0]])
            asc_rank_1 = set(rank_A[0].split('; ')) if isinstance(rank_A[0], str) else set([rank_A[0]])
            stable_best = desc_rank_1.intersection(asc_rank_1)
            
            print("\nСтабильно лучшие альтернативы (находятся в первом ранге обоих дистилляций):")
            print(list(stable_best))
            
        except Exception as e:
            print(f"Ошибка при выполнении ELECTRE III: {e}")
            print("Убедитесь, что модуль algorithm.e_iii доступен и корректно работает.")

    def visualize_all_matrices(self):
        """
        Визуализирует все матрицы парных сравнений и сохраняет графики
        """
        if self.criteria_matrix is None or not self.alt_matrices:
            raise ValueError("Необходимо сначала выполнить анализ критериев и альтернатив")
        
        # Визуализация матрицы критериев
        self._visualize_matrix(self.criteria_matrix, 
                              title="Матрица парных сравнений критериев",
                              row_labels=self.criteria_names,
                              col_labels=self.criteria_names,
                              filename=os.path.join(self.output_dir, "criteria_matrix.png"))
        
        # Визуализация матриц альтернатив
        for i, matrix in enumerate(self.alt_matrices):
            self._visualize_matrix(matrix, 
                                  title=f"Матрица парных сравнений альтернатив по критерию '{self.criteria_names[i]}' (тип: {self.criteria_types[i]})",
                                  row_labels=self.alternative_names,
                                  col_labels=self.alternative_names,
                                  filename=os.path.join(self.output_dir, f"alternatives_matrix_{i+1}.png"))

    def visualize_weights(self):
        """
        Визуализирует веса критериев и альтернатив и сохраняет графики
        """
        if self.criteria_weights is None or not self.alt_weights:
            raise ValueError("Необходимо сначала выполнить анализ критериев и альтернатив")
        
        # Визуализация весов критериев
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.criteria_names, self.criteria_weights)
        
        # Добавляем тип критерия к названию
        labels = [f"{name}\n({ctype})" for name, ctype in zip(self.criteria_names, self.criteria_types)]
        plt.xticks(range(len(self.criteria_names)), labels, rotation=45, ha='right')
        
        plt.ylabel('Вес')
        plt.title('Веса критериев')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "criteria_weights.png"))
        plt.close()
        
        # Визуализация весов альтернатив для каждого критерия
        for i, weights in enumerate(self.alt_weights):
            plt.figure(figsize=(10, 6))
            plt.bar(self.alternative_names, weights)
            plt.ylabel('Вес')
            plt.title(f'Веса альтернатив по критерию "{self.criteria_names[i]}" (тип: {self.criteria_types[i]})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"alternative_weights_{i+1}.png"))
            plt.close()

    def visualize_final_priorities(self):
        """
        Визуализирует итоговые приоритеты альтернатив и сохраняет график
        """
        if self.final_priorities is None:
            raise ValueError("Необходимо сначала вычислить приоритеты")
        
        # Визуализация приоритетов
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.alternative_names, self.final_priorities)
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylabel('Приоритет')
        plt.title('Итоговые приоритеты альтернатив (AHP)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(self.final_priorities) * 1.15)  # Добавляем место для подписей
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "final_priorities.png"))
        plt.close()

    def visualize_electre_results(self):
        """
        Визуализирует результаты ELECTRE III и сохраняет графики
        """
        if self.electre_results is None:
            print("ELECTRE III не выполнялся, визуализация невозможна")
            return
        
        # Визуализация матрицы глобальной конкордации
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(self.electre_results['global_concordance'], 
                                index=self.alternative_names, 
                                columns=self.alternative_names), 
                    annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
        plt.title('Матрица глобальной конкордации (ELECTRE III)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "electre_concordance.png"))
        plt.close()
        
        # Визуализация матрицы доверия
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(self.electre_results['credibility'], 
                                index=self.alternative_names, 
                                columns=self.alternative_names), 
                    annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
        plt.title('Матрица доверия (ELECTRE III)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "electre_credibility.png"))
        plt.close()
        
        # Визуализация ранжирования
        plt.figure(figsize=(12, 6))
        
        # Создаем словарь для хранения рангов
        ranks_desc = {}
        ranks_asc = {}
        
        # Заполняем словари для нисходящей и восходящей дистилляций
        for i, rank_group in enumerate(self.electre_results['rank_D']):
            if isinstance(rank_group, str):
                # Если ранг содержит несколько альтернатив, разделенных точкой с запятой
                for alt in rank_group.split('; '):
                    ranks_desc[alt] = i + 1
            else:
                ranks_desc[rank_group] = i + 1
                
        for i, rank_group in enumerate(self.electre_results['rank_A']):
            if isinstance(rank_group, str):
                for alt in rank_group.split('; '):
                    ranks_asc[alt] = i + 1
            else:
                ranks_asc[rank_group] = i + 1
        
        # Создаем DataFrame для визуализации
        data = []
        for alt in self.alternative_names:
            alt_name = f'a{self.alternative_names.index(alt) + 1}'
            desc_rank = ranks_desc.get(alt_name, None)
            asc_rank = ranks_asc.get(alt_name, None)
            if desc_rank is not None:
                data.append({'Альтернатива': alt, 'Ранг': desc_rank, 'Дистилляция': 'Нисходящая'})
            if asc_rank is not None:
                data.append({'Альтернатива': alt, 'Ранг': asc_rank, 'Дистилляция': 'Восходящая'})
        
        df = pd.DataFrame(data)
        
        # Создаем сгруппированную диаграмму
        sns.barplot(x='Альтернатива', y='Ранг', hue='Дистилляция', data=df)
        plt.title('Ранги альтернатив по ELECTRE III')
        plt.ylabel('Ранг (меньше = лучше)')
        plt.legend(title='Дистилляция')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "electre_ranks.png"))
        plt.close()

    def visualize_decision_matrix(self):
        """
        Визуализирует матрицу решения (альтернативы x критерии) и сохраняет график
        """
        if not self.alt_weights:
            raise ValueError("Необходимо сначала выполнить анализ альтернатив")
        
        # Собираем данные для матрицы решения
        decision_matrix = np.array(self.alt_weights).T
        
        # Создаем DataFrame для визуализации
        df = pd.DataFrame(decision_matrix, 
                          index=self.alternative_names, 
                          columns=[f"{name} ({ctype})" for name, ctype in zip(self.criteria_names, self.criteria_types)])
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
        plt.title('Матрица решения (веса альтернатив по критериям)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "decision_matrix.png"))
        plt.close()

    def visualize_sensitivity(self):
        """
        Визуализирует результаты анализа чувствительности и сохраняет график
        """
        if not self.sensitivity_results:
            raise ValueError("Необходимо сначала выполнить анализ чувствительности")
        
        # Создаем DataFrame для удобства визуализации
        data = []
        for result in self.sensitivity_results:
            for i, alt_name in enumerate(self.alternative_names):
                data.append({
                    'Сценарий': result['name'],
                    'Альтернатива': alt_name,
                    'Приоритет': result['priorities'][i]
                })
        
        df = pd.DataFrame(data)
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        # Создаем уникальные цвета для каждой альтернативы
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.alternative_names)))
        
        for i, alt_name in enumerate(self.alternative_names):
            alt_data = df[df['Альтернатива'] == alt_name]
            plt.plot(alt_data['Сценарий'], alt_data['Приоритет'], 
                     marker='o', linestyle='-', label=alt_name, color=colors[i])
        
        plt.ylabel('Приоритет')
        plt.title('Анализ чувствительности')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensitivity_analysis.png"))
        plt.close()

    def create_summary_dashboard(self):
        """
        Создает сводную инфографику с результатами анализа
        """
        if self.final_priorities is None:
            raise ValueError("Необходимо сначала вычислить приоритеты")
        
        # Настройка размера и сетки графика в зависимости от наличия данных ELECTRE
        has_electre = self.electre_results is not None
        
        # Создаем фигуру с подграфиками
        if has_electre:
            fig = plt.figure(figsize=(20, 18))
            gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1.5, 1.5])
        else:
            fig = plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5])
        
        # 1. Веса критериев
        ax1 = plt.subplot(gs[0, 0])
        bars = ax1.bar(self.criteria_names, self.criteria_weights)
        ax1.set_title('Веса критериев', fontsize=14)
        ax1.set_ylabel('Вес')
        ax1.set_ylim(0, max(self.criteria_weights) * 1.15)
        
        # Добавляем тип критерия к названию
        labels = [f"{name}\n({ctype})" for name, ctype in zip(self.criteria_names, self.criteria_types)]
        ax1.set_xticks(range(len(self.criteria_names)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Итоговые приоритеты AHP
        ax2 = plt.subplot(gs[0, 1])
        bars = ax2.bar(self.alternative_names, self.final_priorities, color='green')
        ax2.set_title('Итоговые приоритеты альтернатив (AHP)', fontsize=14)
        ax2.set_ylabel('Приоритет')
        ax2.set_ylim(0, max(self.final_priorities) * 1.15)
        ax2.tick_params(axis='x', rotation=45)
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Матрица решения
        ax3 = plt.subplot(gs[1, :])
        decision_matrix = np.array(self.alt_weights).T
        df = pd.DataFrame(decision_matrix, 
                          index=self.alternative_names, 
                          columns=[f"{name} ({ctype})" for name, ctype in zip(self.criteria_names, self.criteria_types)])
        
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5, ax=ax3)
        ax3.set_title('Матрица решения (веса альтернатив по критериям)', fontsize=14)
        
        # 4. Анализ чувствительности
        if self.sensitivity_results:
            ax4 = plt.subplot(gs[2, :])
            
            # Создаем DataFrame для анализа чувствительности
            data = []
            for result in self.sensitivity_results:
                for i, alt_name in enumerate(self.alternative_names):
                    data.append({
                        'Сценарий': result['name'],
                        'Альтернатива': alt_name,
                        'Приоритет': result['priorities'][i]
                    })
            
            df_sens = pd.DataFrame(data)
            
            # Визуализация чувствительности
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.alternative_names)))
            
            for i, alt_name in enumerate(self.alternative_names):
                alt_data = df_sens[df_sens['Альтернатива'] == alt_name]
                ax4.plot(alt_data['Сценарий'], alt_data['Приоритет'], 
                        marker='o', linestyle='-', label=alt_name, color=colors[i])
            
            ax4.set_ylabel('Приоритет')
            ax4.set_title('Анализ чувствительности', fontsize=14)
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 5. Если есть данные ELECTRE, добавляем визуализацию рангов
        if has_electre:
            ax5 = plt.subplot(gs[3, :])
            
            # Создаем словари для хранения рангов
            ranks_desc = {}
            ranks_asc = {}
            
            # Заполняем словари для нисходящей и восходящей дистилляций
            for i, rank_group in enumerate(self.electre_results['rank_D']):
                if isinstance(rank_group, str):
                    for alt in rank_group.split('; '):
                        ranks_desc[alt] = i + 1
                else:
                    ranks_desc[rank_group] = i + 1
                    
            for i, rank_group in enumerate(self.electre_results['rank_A']):
                if isinstance(rank_group, str):
                    for alt in rank_group.split('; '):
                        ranks_asc[alt] = i + 1
                else:
                    ranks_asc[rank_group] = i + 1
            
            # Создаем DataFrame для визуализации
            data = []
            for alt in self.alternative_names:
                alt_idx = self.alternative_names.index(alt) + 1
                alt_name = f'a{alt_idx}'
                desc_rank = ranks_desc.get(alt_name, 0)
                asc_rank = ranks_asc.get(alt_name, 0)
                data.append({'alt': alt, 'Нисходящая': desc_rank, 'Восходящая': asc_rank})
            
            electre_df = pd.DataFrame(data)
            electre_df.set_index('alt', inplace=True)
            
            # Строим сгруппированную гистограмму
            electre_df[['Нисходящая', 'Восходящая']].plot(kind='bar', ax=ax5)
            
            ax5.set_title('Ранги альтернатив по ELECTRE III (меньше = лучше)', fontsize=14)
            ax5.set_ylabel('Ранг')
            ax5.set_xlabel('Альтернатива')
            ax5.legend(title='Дистилляция')
            ax5.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "complete_analysis_dashboard.png"), dpi=150)
        plt.close()

    def save_results_to_json(self, filename='ahp_electre_results.json'):
        """
        Сохраняет все результаты анализа в JSON-файл
        
        Parameters:
        -----------
        filename : str
            Имя файла для сохранения
        """
        results = {
            'criteria': {
                'names': self.criteria_names,
                'types': self.criteria_types,
                'matrix': self.criteria_matrix.tolist() if self.criteria_matrix is not None else None,
                'weights': self.criteria_weights.tolist() if self.criteria_weights is not None else None,
                'consistency_ratio': self.criteria_consistency_ratio
            },
            'alternatives': {
                'names': self.alternative_names,
                'matrices': [matrix.tolist() for matrix in self.alt_matrices] if self.alt_matrices else None,
                'weights': [weights.tolist() for weights in self.alt_weights] if self.alt_weights else None,
                'consistency_ratios': self.alt_consistency_ratios
            },
            'ahp_final_priorities': self.final_priorities.tolist() if self.final_priorities is not None else None,
            'ahp_best_alternative': {
                'name': self.alternative_names[np.argmax(self.final_priorities)] if self.final_priorities is not None else None,
                'priority': float(np.max(self.final_priorities)) if self.final_priorities is not None else None
            },
            'sensitivity_analysis': [{
                'name': result['name'],
                'criteria_weights': result['criteria_weights'].tolist(),
                'priorities': result['priorities'].tolist(),
                'best_alternative': {
                    'name': self.alternative_names[np.argmax(result['priorities'])],
                    'priority': float(np.max(result['priorities']))
                }
            } for result in self.sensitivity_results] if self.sensitivity_results else None
        }
        
        # Добавляем результаты ELECTRE III, если они есть
        if self.electre_results:
            results['electre_iii'] = {
                'thresholds': {
                    'Q': self.electre_thresholds['Q'].tolist(),
                    'P': self.electre_thresholds['P'].tolist(),
                    'V': self.electre_thresholds['V'].tolist()
                },
                'global_concordance': self.electre_results['global_concordance'].tolist(),
                'credibility': self.electre_results['credibility'].tolist(),
                'rank_D': self.electre_results['rank_D'],
                'rank_A': self.electre_results['rank_A'],
                'rank_M': self.electre_results['rank_M'],
                'rank_P': self.electre_results['rank_P'].tolist() if isinstance(self.electre_results['rank_P'], np.ndarray) else None
            }
        
        # Сохраняем результаты
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"\nРезультаты успешно сохранены в файл {filepath}")
        
        # Также сохраняем в CSV для удобства
        filepath_csv = os.path.join(self.output_dir, 'final_priorities.csv')
        
        with open(filepath_csv, 'w', encoding='utf-8') as f:
            f.write("Альтернатива,AHP Приоритет")
            if self.electre_results:
                f.write(",ELECTRE III Ранг (нисходящий),ELECTRE III Ранг (восходящий)")
            f.write("\n")
            
            for i, name in enumerate(self.alternative_names):
                line = f"{name},{self.final_priorities[i]:.6f}"
                
                if self.electre_results:
                    # Находим ранг в нисходящей дистилляции
                    alt_name = f'a{i+1}'
                    desc_rank = None
                    asc_rank = None
                    
                    for rank_idx, rank_group in enumerate(self.electre_results['rank_D']):
                        if isinstance(rank_group, str) and alt_name in rank_group.split('; '):
                            desc_rank = rank_idx + 1
                            break
                        elif rank_group == alt_name:
                            desc_rank = rank_idx + 1
                            break
                    
                    # Находим ранг в восходящей дистилляции
                    for rank_idx, rank_group in enumerate(self.electre_results['rank_A']):
                        if isinstance(rank_group, str) and alt_name in rank_group.split('; '):
                            asc_rank = rank_idx + 1
                            break
                        elif rank_group == alt_name:
                            asc_rank = rank_idx + 1
                            break
                    
                    line += f",{desc_rank if desc_rank is not None else 'N/A'},{asc_rank if asc_rank is not None else 'N/A'}"
                
                f.write(line + "\n")
        
        print(f"Итоговые приоритеты сохранены в CSV: {filepath_csv}")

    def _print_matrix(self, matrix, row_labels, col_labels):
        """
        Выводит матрицу в консоль в удобочитаемом формате
        """
        # Подготавливаем данные для вывода
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        print(df.round(4))

    def _visualize_matrix(self, matrix, title, row_labels, col_labels, filename):
        """
        Визуализирует матрицу и сохраняет график
        """
        plt.figure(figsize=(max(8, len(col_labels) * 0.8), max(6, len(row_labels) * 0.8)))
        
        # Создаем DataFrame для удобства визуализации
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        
        # Создаем пользовательскую цветовую карту от светло-голубого до темно-синего
        cmap = LinearSegmentedColormap.from_list('custom_blue', 
                                                ['#EAF2F8', '#2E86C1'], 
                                                N=256)
        
        # Визуализация матрицы
        sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, linewidths=.5, vmin=min(0.1, np.min(matrix)), vmax=max(10, np.max(matrix)))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()


def run_ahp_electre_example():
    """
    Выполняет полный анализ по методам AHP и ELECTRE III с использованием заданных входных данных
    """
    # Создаем директорию для результатов
    output_dir = 'ahp_electre_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Инициализируем анализатор
    ahp = AHPAnalyzer(output_dir=output_dir)
    
    # Количество критериев
    num_criteria = 4
    
    # Количество альтернатив
    num_alternatives = 5
    
    # Ваши данные - формируем матрицу парных сравнений критериев
    criteria_matrix = np.ones((num_criteria, num_criteria))
    
    # Заполняем верхний треугольник матрицы критериев
    criteria_matrix[0][1] = 3
    criteria_matrix[0][2] = 5
    criteria_matrix[0][3] = 1
    criteria_matrix[1][2] = 2
    criteria_matrix[1][3] = 0.5
    criteria_matrix[2][3] = 0.25
    
    # Заполняем нижний треугольник (обратные значения)
    for i in range(num_criteria):
        for j in range(i):
            criteria_matrix[i][j] = 1.0 / criteria_matrix[j][i]
    
    # Анализ критериев (предположим, что первые два критерия на максимизацию, остальные на минимизацию)
    criteria_names = [f'Критерий {i+1}' for i in range(num_criteria)]
    criteria_types = ['max', 'max', 'min', 'min']  # Пример типов критериев
    
    ahp.analyze_criteria(criteria_matrix, criteria_names, criteria_types, method='geometric')
    
    # Создаем матрицы парных сравнений альтернатив по каждому критерию
    alt_matrices = []
    
    # Матрица альтернатив для первого критерия
    alt_matrix_1 = np.ones((num_alternatives, num_alternatives))
    alt_matrix_1[0][1] = 0.5
    alt_matrix_1[0][2] = 0.5
    alt_matrix_1[0][3] = 2
    alt_matrix_1[0][4] = 5
    alt_matrix_1[1][2] = 0.33
    alt_matrix_1[1][3] = 2
    alt_matrix_1[1][4] = 0.33
    alt_matrix_1[2][3] = 4
    alt_matrix_1[2][4] = 2
    alt_matrix_1[3][4] = 1/3
    
    # Заполняем нижний треугольник
    for i in range(num_alternatives):
        for j in range(i):
            alt_matrix_1[i][j] = 1.0 / alt_matrix_1[j][i]
    
    # Матрица альтернатив для второго критерия
    alt_matrix_2 = np.ones((num_alternatives, num_alternatives))
    alt_matrix_2[0][1] = 2
    alt_matrix_2[0][2] = 3
    alt_matrix_2[0][3] = 0.33
    alt_matrix_2[0][4] = 0.5
    alt_matrix_2[1][2] = 0.33
    alt_matrix_2[1][3] = 0.25
    alt_matrix_2[1][4] = 2
    alt_matrix_2[2][3] = 1
    alt_matrix_2[2][4] = 0.33
    alt_matrix_2[3][4] = 0.5
    
    # Заполняем нижний треугольник
    for i in range(num_alternatives):
        for j in range(i):
            alt_matrix_2[i][j] = 1.0 / alt_matrix_2[j][i]
    
    # Матрица альтернатив для третьего критерия
    alt_matrix_3 = np.ones((num_alternatives, num_alternatives))
    alt_matrix_3[0][1] = 0.5
    alt_matrix_3[0][2] = 0.33
    alt_matrix_3[0][3] = 3
    alt_matrix_3[0][4] = 5
    alt_matrix_3[1][2] = 4
    alt_matrix_3[1][3] = 4
    alt_matrix_3[1][4] = 3
    alt_matrix_3[2][3] = 0.5
    alt_matrix_3[2][4] = 2
    alt_matrix_3[3][4] = 0.5
    
    # Заполняем нижний треугольник
    for i in range(num_alternatives):
        for j in range(i):
            alt_matrix_3[i][j] = 1.0 / alt_matrix_3[j][i]
    
    # Матрица альтернатив для четвертого критерия
    alt_matrix_4 = np.ones((num_alternatives, num_alternatives))
    alt_matrix_4[0][1] = 0.5
    alt_matrix_4[0][2] = 2
    alt_matrix_4[0][3] = 0.33
    alt_matrix_4[0][4] = 0.5
    alt_matrix_4[1][2] = 0.5
    alt_matrix_4[1][3] = 0.2
    alt_matrix_4[1][4] = 2
    alt_matrix_4[2][3] = 2
    alt_matrix_4[2][4] = 0.33
    alt_matrix_4[3][4] = 0.5
    
    # Заполняем нижний треугольник
    for i in range(num_alternatives):
        for j in range(i):
            alt_matrix_4[i][j] = 1.0 / alt_matrix_4[j][i]
    
    # Добавляем все матрицы в список
    alt_matrices = [alt_matrix_1, alt_matrix_2, alt_matrix_3, alt_matrix_4]
    
    # Анализ альтернатив
    alternative_names = [f'Альтернатива {i+1}' for i in range(num_alternatives)]
    ahp.analyze_alternatives(alt_matrices, alternative_names, method='geometric')
    
    # Вычисляем итоговые приоритеты по AHP
    ahp.calculate_priorities()
    
    # Проводим анализ чувствительности
    ahp.sensitivity_analysis(variations=0.2, steps=5)
    
    # Запускаем ELECTRE III
    try:
        ahp.run_electre_iii()
    except Exception as e:
        print(f"Ошибка при выполнении ELECTRE III: {e}")
    
    # Визуализация
    ahp.visualize_all_matrices()
    ahp.visualize_weights()
    ahp.visualize_final_priorities()
    ahp.visualize_decision_matrix()
    ahp.visualize_sensitivity()
    
    # Если ELECTRE III выполнен успешно, визуализируем его результаты
    if ahp.electre_results:
        ahp.visualize_electre_results()
    
    # Создаем сводную инфографику
    ahp.create_summary_dashboard()
    
    # Сохраняем все результаты в JSON
    ahp.save_results_to_json()
    
    print(f"\nАнализ завершен. Все результаты сохранены в директории: {output_dir}")

# Запуск примера с указанными данными
if __name__ == "__main__":
    run_ahp_electre_example()