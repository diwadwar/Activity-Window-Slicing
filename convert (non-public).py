import scipy.io
import numpy as np
import os

def convert_mat_to_npz(mat_path, output_path):
    if not os.path.exists(mat_path):
        print(f"Błąd: Nie znaleziono pliku {mat_path}")
        return

    print(f"Wczytywanie {mat_path}...")
    try:
        mat = scipy.io.loadmat(mat_path)
        data_variable = mat['data']
        
        all_subsets_samples = []
        all_subsets_labels = []
        
        num_subsets = data_variable.size
        print(f"Znaleziono {num_subsets} podzbiorów. Rozpoczynam przetwarzanie...")
        
        # Iteracja po każdym podzbiorze
        for subset_idx, subset in enumerate(data_variable.flatten()):
            raw_samples = subset.T[0] 
            raw_labels = subset.T[1]

            subset_samples = []
            subset_labels = []
            
            print(f"Przetwarzanie podzbioru {subset_idx} (próbki: {len(raw_samples)})...")
            
            for sample in raw_samples:
                arr = np.array(sample)
                                    
                subset_samples.append(arr.T)

            for label in raw_labels:
                subset_labels.append(label.item())

            # Bezpieczne tworzenie tablicy próbek dla TEGO podzbioru
            subset_array = np.empty(len(subset_samples), dtype=object)
            for i, arr in enumerate(subset_samples):
                subset_array[i] = arr
                
            all_subsets_samples.append(subset_array)
            all_subsets_labels.append(np.array(subset_labels))

        print("-" * 30)
        print(f"Liczba zachowanych podzbiorów: {len(all_subsets_samples)}")
        print("-" * 30)

        # Bezpieczne tworzenie GŁÓWNEJ tablicy podzbiorów
        data_array = np.empty(len(all_subsets_samples), dtype=object)
        for i, arr in enumerate(all_subsets_samples):
            data_array[i] = arr
            
        labels_array = np.empty(len(all_subsets_labels), dtype=object)
        for i, arr in enumerate(all_subsets_labels):
            labels_array[i] = arr

        print(f"Zapisywanie do {output_path}...")
        np.savez(output_path, data=data_array, labels=labels_array)
        print("Gotowe!")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_mat_to_npz('ECG.mat', 'ECG.npz')