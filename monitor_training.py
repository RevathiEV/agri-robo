"""
Quick script to monitor training progress by checking model file updates
"""
import os
from datetime import datetime
import time

model_files = [
    'tomato_disease_model.h5',
    'tomato_disease_model_best.h5'
]

print("Monitoring training progress...")
print("Press Ctrl+C to stop monitoring\n")

try:
    last_sizes = {}
    for f in model_files:
        if os.path.exists(f):
            last_sizes[f] = os.path.getsize(f)
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking model files...")
        for model_file in model_files:
            if os.path.exists(model_file):
                mod_time = os.path.getmtime(model_file)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                
                # Check if file was updated
                if model_file in last_sizes:
                    size_changed = file_size != (last_sizes[model_file] / (1024 * 1024))
                    status = "[UPDATING]" if size_changed else "[IDLE]"
                else:
                    status = "[NEW]"
                
                print(f"  {model_file}: {file_size:.2f} MB - {status} - Last: {mod_time_str}")
                last_sizes[model_file] = os.path.getsize(model_file)
            else:
                print(f"  {model_file}: Not found yet")
        
        time.sleep(30)  # Check every 30 seconds
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")

