import csv
import os
from typing import Dict, List, Union

def load_historical_demands(data_dir: str, filename: str = "demand_history.csv") -> Dict[int, List[float]]:
    """
    Loads historical demand data from a CSV file.
    
    Expected CSV format:
    dish_id,day,demand
    0,1,12.5
    0,2,14.0
    1,1,30.0
    ...
    
    Args:
        data_dir: Directory containing the CSV file
        filename: Name of the CSV file
        
    Returns:
        Dict mapping dish_id (int) to list of demand values (float)
        sorted by day (assuming CSV is sorted or days are sequential)
    """
    file_path = os.path.join(data_dir, filename)
    history: Dict[int, List[float]] = {}
    
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found. Proceeding without historical data.")
        return history
        
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Collect data
            temp_data = [] 
            for row in reader:
                # Support various column names
                dish_key = 'dish_id' if 'dish_id' in row else 'dish_name'
                day_key = 'day' if 'day' in row else 'date'
                demand_key = 'demand' if 'demand' in row else 'qty'
                
                if dish_key not in row or demand_key not in row:
                    continue
                    
                try:
                    d_id = int(row[dish_key])
                    day = int(row[day_key]) if day_key in row else 0
                    qty = float(row[demand_key])
                    temp_data.append({'id': d_id, 'day': day, 'qty': qty})
                except ValueError:
                    continue
            
            # Sort by day and group by dish_id
            temp_data.sort(key=lambda x: x['day'])
            
            for item in temp_data:
                d_id = item['id']
                if d_id not in history:
                    history[d_id] = []
                history[d_id].append(item['qty'])
                
    except Exception as e:
        print(f"Error loading demand data: {e}")
        
    return history
