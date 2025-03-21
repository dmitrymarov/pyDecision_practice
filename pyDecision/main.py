import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import json
import os
from algorithm.e_iii import electre_iii

SSI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

def electre_iii_fixed(dataset, P, Q, V, W, graph=False):
    alts   = list(range(1, dataset.shape[0] + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    alts_D = [0]*dataset.shape[0]
    alts_A = [0]*dataset.shape[0]
    # Call the original function's utility functions but save the graph instead of showing it
    from algorithm.e_iii import global_concordance_matrix, credibility_matrix, destilation_descending, destilation_ascending, pre_order_matrix
    
    global_concordance = global_concordance_matrix(dataset, P=P, Q=Q, W=W)
    credibility = credibility_matrix(dataset, global_concordance, P=P, V=V)
    rank_D = destilation_descending(credibility=credibility)
    rank_A = destilation_ascending(credibility=credibility)
    rank_M = []
    for i in range(0, dataset.shape[0]):
        for j in range(0, len(rank_D)):
            if (alts[i] in rank_D[j]):
                alts_D[i] = j + 1
        for k in range(0, len(rank_A)):
            if (alts[i] in rank_A[k]):
                alts_A[i] = k + 1 
    for i in range(0, len(alts)):
        rank_M.append('a' + str(i+1))
    rank_M.sort()
    rank_P = pre_order_matrix(rank_D, rank_A, number_of_alternatives=dataset.shape[0])
    if (graph == True):
        po_ranking_fixed(rank_P)
    return global_concordance, credibility, rank_D, rank_A, rank_M, rank_P

def matrix(number):
    A = np.ones([number, number])
    for i in range(0, number):
        for j in range(0, number):
            if i < j:
                a = str(
                    input(
                        f'Насколько предпочтительней {i+1} элемент множества чем {j+1} элемент множества?'
                        + ' Введите рациональное положительное число: '))
                ratPos = a.replace(',', '.')
                A[i, j] = float(ratPos)
                A[j, i] = 1 / float(ratPos)
    return A

def vector(matrix):
    matSum = np.zeros(len(matrix))
    for i in range(0, len(matrix)):
        vect = matrix[:, i]
        w = vect / vect.sum()
        for j in range(0, len(matrix)):
            matSum[j] += w[j]
    matSum = matSum / len(matrix)
    return matSum

def zVector(number, store):
    temp = 0
    for i in range(0, int(number)):
        temp = temp + np.matmul(store[0][i], store[1]) / store[1][i]
    
    lmax = np.round(temp / int(number), 3)
    print(f'\n { lmax } - собственное значение МПС')
    IS = np.round((lmax - int(number)) / (int(number) - 1), 3)
    print(f' { IS } - индекса согласованности (ИС) МПС')
    OS = np.round(IS / SSI[int(number) - 1], 4)
    print(f' { OS } - отношение согласованности (ОС) МПС')
    
    return OS

def get_default_thresholds(performance_matrix, criterion_type):
    """Автоматически определяет разумные пороговые значения на основе данных"""
    ranges = np.max(performance_matrix, axis=0) - np.min(performance_matrix, axis=0)
    
    # Защита от нулевого диапазона
    ranges = np.where(ranges == 0, 0.01, ranges)
    
    Q = ranges * 0.05  # 5% диапазона для порога безразличия
    P = ranges * 0.2   # 20% диапазона для порога предпочтения
    V = ranges * 0.5   # 50% диапазона для порога вето
    
    return Q, P, V

def sensitivity_analysis(performance_matrix, weights, criterion_type, Q, P, V):
    """Проводит анализ чувствительности, изменяя веса и пороги"""
    results = []
    
    # Исходный результат
    global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(
        dataset=performance_matrix, 
        P=P, 
        Q=Q, 
        V=V, 
        W=weights,
        graph=False
    )
    results.append(("Исходные параметры", rank_D))
    
    # Вариации весов
    for i in range(len(weights)):
        # Увеличение веса одного критерия на 20%
        adjusted_weights = weights.copy()
        if adjusted_weights[i] * 1.2 <= 1.0:  # Проверка на допустимость увеличения
            adjusted_weights[i] *= 1.2
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)  # Ренормализация
            
            global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(
                dataset=performance_matrix, 
                P=P, 
                Q=Q, 
                V=V, 
                W=adjusted_weights,
                graph=False
            )
            results.append((f"Увеличение веса критерия {i+1} на 20%", rank_D))
    
    # Вариации порогов
    # Увеличение всех порогов на 20%
    global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(
        dataset=performance_matrix, 
        P=np.array(P) * 1.2, 
        Q=np.array(Q) * 1.2, 
        V=np.array(V) * 1.2, 
        W=weights,
        graph=False
    )
    results.append(("Увеличение всех порогов на 20%", rank_D))
    
    # Уменьшение всех порогов на 20%
    global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii(
        dataset=performance_matrix, 
        P=np.array(P) * 0.8, 
        Q=np.array(Q) * 0.8, 
        V=np.array(V) * 0.8, 
        W=weights,
        graph=False
    )
    results.append(("Уменьшение всех порогов на 20%", rank_D))
    
    return results

def visualize_sensitivity(sensitivity_results, numberAlt):
    """Визуализирует результаты анализа чувствительности"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Подготовка данных для визуализации
    scenarios = [result[0] for result in sensitivity_results]
    alternative_ranks = {}
    
    # Инициализация словаря рангов
    for i in range(1, numberAlt + 1):
        alternative_ranks[f'a{i}'] = []
    
    # Заполнение рангов для каждого сценария
    for scenario, ranks in sensitivity_results:
        # Для каждой альтернативы найдем ее ранг
        for alt in alternative_ranks.keys():
            rank_found = False
            for rank_idx, rank_group in enumerate(ranks):
                if alt in rank_group:
                    alternative_ranks[alt].append(rank_idx + 1)
                    rank_found = True
                    break
                # Handle semicolon separated alternatives
                elif isinstance(rank_group, str) and alt in rank_group.split(';'):
                    alternative_ranks[alt].append(rank_idx + 1)
                    rank_found = True
                    break
            if not rank_found:
                alternative_ranks[alt].append(0)  # Если не найдено, присваиваем 0
    
    # Построение графика
    x = np.arange(len(scenarios))
    width = 0.8 / len(alternative_ranks)
    
    for i, (alt, ranks) in enumerate(alternative_ranks.items()):
        ax.bar(x + i * width - 0.4 + width/2, ranks, width, label=alt)
    
    ax.set_ylabel('Ранг (меньше = лучше)')
    ax.set_title('Анализ чувствительности рангов альтернатив')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    # Save figure instead of showing it
    plt.savefig('sensitivity_analysis.png')
    plt.close()
    
    return alternative_ranks

def po_ranking_fixed(po_string):
    alts = list(range(1, po_string.shape[0] + 1)) 
    alts = ['a' + str(alt) for alt in alts]
    for i in range (po_string.shape[0] - 1, -1, -1):
        for j in range (po_string.shape[1] -1, -1, -1):
            if (po_string[i,j] == 'I'):
                po_string = np.delete(po_string, i, axis = 0)
                po_string = np.delete(po_string, i, axis = 1)
                alts[j] = str(alts[j] + "; " + alts[i])
                del alts[i]
                break    
    graph = {}
    for i in range(po_string.shape[0]):
        if (len(alts[i]) == 0):
            graph[alts[i]] = i 
        else:
            graph[alts[i][ :2]] = i   
            graph[alts[i][-2:]] = i 
    po_matrix = np.zeros((po_string.shape[0], po_string.shape[1]))
    for i in range (0, po_string.shape[0]):
        for j in range (0, po_string.shape[1]):
            if (po_string[i,j] == 'P+'):
                po_matrix[i,j] = 1
    col_sum = np.sum(po_matrix, axis = 1)
    alts_rank = [x for _, x in sorted(zip(col_sum, alts))]
    if (np.sum(col_sum) != 0):
        alts_rank.reverse()      
    graph_rank = {}
    for i in range(po_string.shape[0]):
        if (len(alts_rank[i]) == 0):
            graph_rank[alts_rank[i]] = i 
        else:
            graph_rank[alts_rank[i][ :2]] = i   
            graph_rank[alts_rank[i][-2:]] = i
    rank = np.copy(po_matrix)
    for i in range(0, po_matrix.shape[0]):
        for j in range(0, po_matrix.shape[1]): 
            if (po_matrix[i,j] == 1):
                rank[i,:] = np.clip(rank[i,:] - rank[j,:], 0, 1)   
    rank_xy = np.zeros((len(alts_rank), 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        if (len(alts_rank) - np.sum(~rank.any(1)) != 0):
            rank_xy[i, 1] = len(alts_rank) - np.sum(~rank.any(1))
        else:
            rank_xy[i, 1] = 1
    for i in range(0, len(alts_rank) - 1):
        i1 = int(graph[alts_rank[ i ][:2]]) 
        i2 = int(graph[alts_rank[i+1][:2]])
        if (po_string[i1,i2] == 'P+'):
            rank_xy[i+1,1] = rank_xy[i+1,1] - 1
            for j in range(i+2, rank_xy.shape[0]):
                rank_xy[j,1] = rank_xy[i+1,1]
        if (po_string[i1,i2] == 'R'):
            rank_xy[i+1,0] = rank_xy[i,0] + 1            
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], alts_rank[i], size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, len(alts_rank)):
        alts_rank[i] = alts_rank[i][:2]
    for i in range(0, rank.shape[0]):
        for j in range(0, rank.shape[1]):
            k1 = int(graph_rank[list(graph.keys())[list(graph.values()).index(i)]])
            k2 = int(graph_rank[list(graph.keys())[list(graph.values()).index(j)]])
            if (rank[i, j] == 1):  
                plt.arrow(rank_xy[k1, 0], rank_xy[k1, 1], rank_xy[k2, 0] - rank_xy[k1, 0], rank_xy[k2, 1] - rank_xy[k1, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    xmin = np.amin(rank_xy[:,0])
    xmax = np.amax(rank_xy[:,0])
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    # Save figure instead of showing it
    plt.savefig('po_ranking.png')
    plt.close()
    return

def save_data(data, filename):
    """Сохраняет данные в JSON-файл"""
    try:
        # Преобразуем numpy массивы в списки для JSON-сериализации
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                json_data[key] = [item.tolist() for item in value]
            elif key == 'electre_results':
                # Handle nested dictionaries with numpy arrays
                json_data[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_data[key][subkey] = subvalue.tolist()
                    else:
                        json_data[key][subkey] = subvalue
            else:
                json_data[key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Данные успешно сохранены в файл {filename}")
        return True
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")
        return False


def load_data(filename):
    """Загружает данные из JSON-файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Преобразуем списки обратно в numpy массивы
        for key, value in data.items():
            if isinstance(value, list):
                if key in ['A', 'weights', 'performance_matrix', 'Q', 'P', 'V']:
                    data[key] = np.array(value)
        
        print(f"Данные успешно загружены из файла {filename}")
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def display_results(data, sensitivity_analysis_results=None):
    """Отображает результаты анализа"""
    print("\n===== РЕЗУЛЬТАТЫ АНАЛИЗА =====")
    
    # AHP результаты
    print("\nВеса критериев (AHP):")
    for i, w in enumerate(data['weights']):
        print(f"Критерий {i+1}: {w:.4f}")
    
    # AHP ранжирование
    print("\nРезультаты AHP:")
    for i, result in enumerate(data['ahp_result']):
        print(f"Альтернатива {i+1}: {result:.4f}")
    
    # ELECTRE III результаты
    if 'electre_results' in data:
        print("\nРезультаты ELECTRE III:")
        print("Ранжирование при нисходящей дистилляции (Descending Distillation):")
        for i, rank_group in enumerate(data['electre_results']['rank_D']):
            print(f"Ранг {i+1}: {rank_group}")
            
        print("\nРанжирование при восходящей дистилляции (Ascending Distillation):")
        for i, rank_group in enumerate(data['electre_results']['rank_A']):
            print(f"Ранг {i+1}: {rank_group}")
    
    # Результаты анализа чувствительности
    if sensitivity_analysis_results:
        print("\nРезультаты анализа чувствительности:")
        
        # Определение стабильных альтернатив
        first_rank = data['electre_results']['rank_D'][0] if data['electre_results']['rank_D'] else ""
        
        # Parse alternatives from potentially semi-colon separated string
        alternatives = []
        if isinstance(first_rank, str):
            alternatives = [alt.strip() for alt in first_rank.split(';')]
        
        stable_alts = []
        unstable_alts = []
        
        for alt in alternatives:
            if not alt:  # Skip empty alternatives
                continue
                
            stable = True
            for scenario, ranks in sensitivity_analysis_results[1:]:
                first_rank_scenario = ranks[0] if ranks else ""
                scenario_alts = []
                
                if isinstance(first_rank_scenario, str):
                    scenario_alts = [a.strip() for a in first_rank_scenario.split(';')]
                
                if alt not in scenario_alts:
                    stable = False
                    break
            
            if stable:
                stable_alts.append(alt)
            else:
                unstable_alts.append(alt)
        
        print("\nСтабильно лучшие альтернативы (всегда в первом ранге):")
        print(stable_alts)
        
        print("\nНестабильные альтернативы (могут выпадать из первого ранга):")
        print(unstable_alts)



def main():
    # Проверка наличия сохраненных данных
    if os.path.exists('saved_analysis.json'):
        load_option = input("Найдены сохраненные данные. Загрузить их? (да/нет): ").lower()
        if load_option == 'да':
            data = load_data('saved_analysis.json')
            if data:
                display_results(data)
                return
    
    # Phase 1: Use AHP to determine criteria weights
    numberCrit = str(input("Укажите размерность матрицы. Напишите целое число: "))
    if not numberCrit.isdigit():
        print("Введенное значение не int\n")
        return main()
    
    # Step 1: AHP for criteria weights
    A = matrix(int(numberCrit))
    print(A)
    store = []
    store.append(A)
    weights = vector(A)
    store.append(weights)
    print("\nВектор приоритетов для критериев")
    for i in range(len(weights)):
        print(f'Критерий {i+1} = {np.round(weights[i], 3)}')

    OS = zVector(numberCrit, store)
    
    if (OS > 0.1):
        print('\nСогласованность является неприемлемой, пересмотрите матрицу предпочтений!')
        return main()
    
    # Step 2: Get alternatives
    numberAlt = str(input("\nКакое количество альтернатив нужно ввести? Напишите целое число: "))
    if not numberAlt.isdigit():
        print("Введенное значение не int\n")
        return main()
    
    numberAlt = int(numberAlt)
    
    # Step 3: For each criterion, create performance matrix using pairwise comparisons (AHP approach)
    performance_matrix = np.zeros((numberAlt, int(numberCrit)))
    
    # First, we'll use AHP to get pairwise comparisons for each criterion
    for i in range(0, int(numberCrit)):
        print(f'\nДанные для критерия {i+1}:')
        A = matrix(int(numberAlt))
        store1 = []
        store1.append(A)
        print(A)
        performance_matrix[:, i] = vector(A)
        store1.append(performance_matrix[:, i])
        
        OS = zVector(numberAlt, store1)
        if (OS > 0.1):
            print('\nСогласованность является неприемлемой, пересмотрите матрицу предпочтений!')
            return main()
    
    print("\nВекторы приоритетов для альтернатив (Performance Matrix): \n")
    print(performance_matrix)
    
    # Calculate AHP result for comparison
    ahp_result = np.matmul(performance_matrix, weights)
    print(f'\nАНР Вектор приоритетов - { ahp_result }')
    for i in range(len(ahp_result)):
        print(f'Альтернатива {i+1} = {np.round(ahp_result[i], 3)}')
    
    # Phase 2: Use ELECTRE III with the weights from AHP
    # Define criterion types ('max' or 'min')
    criterion_type = []
    print("\nТип критерия (max/min):")
    for i in range(int(numberCrit)):
        while True:
            ctype = input(f"Тип для критерия {i+1} (введите 'max' для максимизации или 'min' для минимизации): ").lower()
            if ctype in ['max', 'min']:
                criterion_type.append(ctype)
                break
            else:
                print("Введите 'max' или 'min'.")
    
    # Автоматическое определение пороговых значений
    default_Q, default_P, default_V = get_default_thresholds(performance_matrix, criterion_type)
    
    print("\nАвтоматически рассчитанные пороговые значения:")
    for i in range(int(numberCrit)):
        print(f"Критерий {i+1}: ")
        print(f"  Q (порог безразличия): {default_Q[i]:.4f}")
        print(f"  P (порог предпочтения): {default_P[i]:.4f}")
        print(f"  V (порог вето): {default_V[i]:.4f}")
    
    use_defaults = input("\nИспользовать автоматически рассчитанные пороговые значения? (да/нет): ").lower()
    
    if use_defaults == 'да':
        Q = default_Q
        P = default_P
        V = default_V
    else:
        # Define thresholds for ELECTRE III
        print("\nОпределите пороговые значения вручную:")
        
        # Indifference threshold (Q)
        Q = []
        print("\nПорог безразличия (Q) - минимальное значение, ниже которого различие несущественно:")
        for i in range(int(numberCrit)):
            Q.append(float(input(f"Q для критерия {i+1} [{default_Q[i]:.4f}]: ").replace(',', '.') or default_Q[i]))
        
        # Preference threshold (P)
        P = []
        print("\nПорог предпочтения (P) - значение, выше которого предпочтение становится строгим:")
        for i in range(int(numberCrit)):
            P.append(float(input(f"P для критерия {i+1} [{default_P[i]:.4f}]: ").replace(',', '.') or default_P[i]))
        
        # Veto threshold (V)
        V = []
        print("\nПорог вето (V) - значение, выше которого вето накладывается на утверждение:")
        for i in range(int(numberCrit)):
            V.append(float(input(f"V для критерия {i+1} [{default_V[i]:.4f}]: ").replace(',', '.') or default_V[i]))
    
    # Apply ELECTRE III
    try:
        global_concordance, credibility, rank_D, rank_A, rank_M, rank_P = electre_iii_fixed(
            dataset=performance_matrix, 
            P=P, 
            Q=Q, 
            V=V, 
            W=weights,
            graph=True  # This will save the pre-order graph instead of showing it
        )
        
        electre_results = {
            'global_concordance': global_concordance,
            'credibility': credibility,
            'rank_D': rank_D,
            'rank_A': rank_A,
            'rank_M': rank_M,
            'rank_P': rank_P
        }
        
        print("\nРезультаты ELECTRE III:")
        print("Ранжирование при нисходящей дистилляции (Descending Distillation):")
        for i, rank_group in enumerate(rank_D):
            print(f"Ранг {i+1}: {rank_group}")
            
        print("\nРанжирование при восходящей дистилляции (Ascending Distillation):")
        for i, rank_group in enumerate(rank_A):
            print(f"Ранг {i+1}: {rank_group}")
        
        # Анализ чувствительности
        do_sensitivity = input("\nВыполнить анализ чувствительности? (да/нет): ").lower()
        sensitivity_results = None
        
        if do_sensitivity == 'да':
            sensitivity_results = sensitivity_analysis(performance_matrix, weights, criterion_type, Q, P, V)
            
            print("\nРезультаты анализа чувствительности:")
            for desc, ranks in sensitivity_results:
                print(f"\n{desc}:")
                for i, rank_group in enumerate(ranks):
                    print(f"Ранг {i+1}: {rank_group}")
            
            # Визуализация результатов анализа чувствительности
            alternative_ranks = visualize_sensitivity(sensitivity_results, numberAlt)
        
        # Сохранение результатов
        data = {
            'A': A,
            'weights': weights,
            'numberCrit': int(numberCrit),
            'numberAlt': numberAlt,
            'performance_matrix': performance_matrix,
            'criterion_type': criterion_type,
            'Q': Q,
            'P': P,
            'V': V,
            'ahp_result': ahp_result,
            'electre_results': electre_results
        }
        
        save_option = input("\nСохранить результаты анализа? (да/нет): ").lower()
        if save_option == 'да':
            filename = input("Введите имя файла [saved_analysis.json]: ") or 'saved_analysis.json'
            save_data(data, filename)
        
        # Отображение всех результатов
        display_results(data, sensitivity_results)
        
    except Exception as e:
        print(f"Ошибка при выполнении ELECTRE III: {e}")
        print("Проверьте входные данные и пороги.")

if __name__ == "__main__":
    main()