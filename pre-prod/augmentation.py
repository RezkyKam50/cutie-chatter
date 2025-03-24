import ollama
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from collections import Counter
import threading
import matplotlib, os
matplotlib.use('Qt5Agg')  
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''

This is used to generate paraphrased text for each row in a dataset

'''


augmentation_counts = {'total': 0, 'label_2': 0, 'label_3': 0, 'label_4': 0, 'label_5': 0}
lock = threading.Lock()   

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Real-time Data Augmentation Progress', fontsize=16)

def update_plot(frame):
    with lock:
        ax1.clear()
        labels = ['Label 2', 'Label 3', 'Label 4', 'Label 5']
        counts = [augmentation_counts['label_2'], augmentation_counts['label_3'], 
                 augmentation_counts['label_4'], augmentation_counts['label_5']]
        if augmentation_counts['total'] > 0:
            percentages = [count / target_counts.get(label, 1) * 100 for label, count in zip([2, 3, 4, 5], counts)]
        else:
            percentages = [0, 0, 0, 0]
            
        colors = ['skyblue', 'lightgreen', 'yellowgreen', 'coral']
        bars = ax1.barh(labels, percentages, color=colors)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Percentage Complete')
        ax1.set_title('Augmentation Progress by Label')

        for i, bar in enumerate(bars):
            if percentages[i] > 0:
                ax1.text(
                    min(percentages[i] + 2, 95), 
                    bar.get_y() + bar.get_height()/2, 
                    f"{percentages[i]:.1f}% ({counts[i]}/{target_counts.get(int(labels[i].split()[-1]), 0)})",
                    va='center'
                )
        ax2.clear()
        all_labels = list(current_distribution.keys())
        all_counts = list(current_distribution.values())
        
        ax2.bar(all_labels, all_counts, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Emotion Label')
        ax2.set_ylabel('Count')
        ax2.set_title('Current Dataset Distribution')

        for i, count in enumerate(all_counts):
            ax2.text(all_labels[i], count + max(all_counts) * 0.02, str(count), ha='center')
            
    return bars,

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("Dataset/Emotion/emodata_augmented.csv")
    original_length = len(df)
    labels_to_augment = [5, 4, 3, 2]
    df_to_augment = df[df['label'].isin(labels_to_augment)]
    label_counts = df['label'].value_counts().to_dict()
    current_distribution = {label: count for label, count in sorted(label_counts.items())}
    target_counts = {
        2: df_to_augment[df_to_augment['label'] == 2].shape[0],
        3: df_to_augment[df_to_augment['label'] == 3].shape[0],
        4: df_to_augment[df_to_augment['label'] == 4].shape[0],
        5: df_to_augment[df_to_augment['label'] == 5].shape[0]
    }
    augmented_df = df.copy()
    ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
    plt.ion() 
    plt.show(block=False)

    try:
        for index, row in tqdm(df_to_augment.iterrows(), total=len(df_to_augment)):
            text = row['text']
            label = row['label']
            print(f"\nProcessing entry {index} | Label: {label}")
            print(f"Original: {text}")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stream = ollama.chat(
                        model="huihui_ai/qwen2.5-1m-abliterated:7b",
                        messages=[
                            {'role': 'system', 'content': "Paraphrase the given text by the user, don't give the user any advice or questions."},
                            {'role': 'user', 'content': f"Paraphrase: {text}"}
                        ],
                        stream=True,
                        options={
                            "num_thread": 8,
                            "temperature": 0.5,
                            "f16_kv": True,
                            "num_ctx": 128,
                            "num_batch": 32,
                            "num_prediction": 128
                        }
                    )
                    
                    full_content = ''
                    for chunk in stream:
                        if chunk and 'message' in chunk and 'content' in chunk['message']:
                            full_content += chunk['message']['content']
                    if full_content.strip():
                        print(f"Augmented: {full_content}")
                        augmented_df.loc[len(augmented_df)] = [full_content, label]
                        with lock:
                            augmentation_counts['total'] += 1
                            augmentation_counts[f'label_{label}'] += 1
                            current_distribution = Counter(augmented_df['label'])
                            current_distribution = {label: count for label, count in sorted(current_distribution.items())}
                        
                        break
                    else:
                        raise ValueError("Empty response received")
                
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt   
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to augment after {max_retries} attempts. Skipping this entry.")
            if (index % 10 == 0 and index > 0) or index == df_to_augment.index[-1]:
                augmented_df.to_csv("Dataset/Emotion/emodata_augmented.csv", index=False)
                print(f"Intermediate save: Dataset updated with {len(augmented_df) - original_length} new entries so far")
            plt.pause(0.1)
        augmented_df.to_csv("Dataset/Emotion/emodata_augmented.csv", index=False)
        print(f"Dataset updated successfully! Added {len(augmented_df) - original_length} new entries for labels 5, 4, 3, and 2.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        augmented_df.to_csv("Dataset/Emotion/emodata_augmented.csv", index=False)
        print(f"Saved {len(augmented_df) - original_length} augmented entries before interruption.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Attempting to save current progress...")
        augmented_df.to_csv("Dataset/Emotion/emodata_recovery.csv", index=False)
        print(f"Emergency save completed to emodata_recovery.csv")
    
    finally:
        print("Augmentation complete or interrupted. Close the plot window to exit.")
        plt.ioff()
        plt.show()
