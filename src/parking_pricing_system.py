# Dynamic Pricing for Urban Parking Lots
# Capstone Project - Summer Analytics 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages (run in Colab)
# !pip install pathway-python bokeh

class ParkingPricingSystem:
    """
    A comprehensive dynamic pricing system for urban parking lots
    Implements three models: Baseline Linear, Demand-Based, and Competitive Pricing
    """
    
    def __init__(self, base_price=10.0):
        self.base_price = base_price
        self.parking_lots = {}
        self.historical_data = {}
        self.competitor_matrix = None
        
    def initialize_parking_lots(self, data):
        """Initialize parking lot information from data"""
        unique_locations = data[['latitude', 'longitude', 'capacity']].drop_duplicates()
        
        for idx, row in unique_locations.iterrows():
            lot_id = f"lot_{idx}"
            self.parking_lots[lot_id] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'capacity': row['capacity'],
                'current_price': self.base_price,
                'price_history': [self.base_price]
            }
        
        # Calculate competitor proximity matrix
        self._calculate_competitor_matrix()
    
    def _calculate_competitor_matrix(self):
        """Calculate distance matrix between parking lots"""
        n_lots = len(self.parking_lots)
        self.competitor_matrix = np.zeros((n_lots, n_lots))
        
        lot_ids = list(self.parking_lots.keys())
        for i, lot_i in enumerate(lot_ids):
            for j, lot_j in enumerate(lot_ids):
                if i != j:
                    dist = self._haversine_distance(
                        self.parking_lots[lot_i]['latitude'],
                        self.parking_lots[lot_i]['longitude'],
                        self.parking_lots[lot_j]['latitude'],
                        self.parking_lots[lot_j]['longitude']
                    )
                    self.competitor_matrix[i][j] = dist
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def model_1_baseline_linear(self, lot_data):
        """
        Model 1: Baseline Linear Pricing
        Price adjustment based on occupancy rate
        """
        occupancy_rate = lot_data['occupancy'] / lot_data['capacity']
        alpha = 0.5  # Sensitivity parameter
        
        # Simple linear adjustment
        price_adjustment = alpha * occupancy_rate
        new_price = max(self.base_price * (1 + price_adjustment), self.base_price * 0.5)
        new_price = min(new_price, self.base_price * 2.0)  # Cap at 2x base price
        
        return new_price
    
    def model_2_demand_based(self, lot_data):
        """
        Model 2: Demand-Based Pricing
        Sophisticated demand function considering multiple factors
        """
        # Normalize features
        occupancy_rate = lot_data['occupancy'] / lot_data['capacity']
        queue_normalized = min(lot_data['queue_length'] / 10, 1.0)  # Normalize queue
        traffic_normalized = lot_data['traffic_level'] / 10.0  # Assume traffic 0-10 scale
        
        # Vehicle type weights
        vehicle_weights = {'car': 1.0, 'bike': 0.5, 'truck': 1.5}
        vehicle_weight = vehicle_weights.get(lot_data['vehicle_type'], 1.0)
        
        # Demand function parameters
        alpha = 0.4  # Occupancy coefficient
        beta = 0.3   # Queue coefficient
        gamma = 0.2  # Traffic coefficient (negative impact)
        delta = 0.3  # Special day coefficient
        epsilon = 0.1 # Vehicle type coefficient
        
        # Calculate demand score
        demand_score = (alpha * occupancy_rate + 
                       beta * queue_normalized - 
                       gamma * traffic_normalized + 
                       delta * lot_data['is_special_day'] + 
                       epsilon * vehicle_weight)
        
        # Normalize demand score to [-1, 1]
        demand_normalized = np.tanh(demand_score)
        
        # Calculate price multiplier
        lambda_param = 0.5  # Price sensitivity
        price_multiplier = 1 + lambda_param * demand_normalized
        
        # Apply bounds
        price_multiplier = max(0.5, min(2.0, price_multiplier))
        
        new_price = self.base_price * price_multiplier
        return new_price
    
    def model_3_competitive_pricing(self, lot_data, lot_id):
        """
        Model 3: Competitive Pricing
        Considers nearby competitor prices and suggests rerouting
        """
        # Start with demand-based price
        base_competitive_price = self.model_2_demand_based(lot_data)
        
        # Get competitor prices within 2km radius
        lot_idx = list(self.parking_lots.keys()).index(lot_id)
        nearby_competitors = []
        
        for i, comp_lot_id in enumerate(self.parking_lots.keys()):
            if i != lot_idx and self.competitor_matrix[lot_idx][i] <= 2.0:  # Within 2km
                nearby_competitors.append(self.parking_lots[comp_lot_id]['current_price'])
        
        if nearby_competitors:
            avg_competitor_price = np.mean(nearby_competitors)
            min_competitor_price = np.min(nearby_competitors)
            
            # Competitive adjustment
            if lot_data['occupancy'] / lot_data['capacity'] > 0.8:  # High occupancy
                if base_competitive_price > avg_competitor_price:
                    # Reduce price to be competitive
                    competitive_adjustment = -0.1 * (base_competitive_price - avg_competitor_price)
                    base_competitive_price += competitive_adjustment
            else:
                # Can price higher if competitors are expensive
                if avg_competitor_price > base_competitive_price:
                    competitive_adjustment = 0.1 * (avg_competitor_price - base_competitive_price)
                    base_competitive_price += competitive_adjustment
        
        # Rerouting suggestion
        rerouting_suggestion = None
        if (lot_data['occupancy'] / lot_data['capacity'] > 0.9 and 
            lot_data['queue_length'] > 5 and nearby_competitors):
            
            cheapest_competitor_idx = np.argmin(nearby_competitors)
            rerouting_suggestion = f"Consider rerouting to nearby lot (Price: ${nearby_competitors[cheapest_competitor_idx]:.2f})"
        
        return base_competitive_price, rerouting_suggestion
    
    def update_prices(self, current_data, model_type='demand_based'):
        """Update prices for all parking lots based on current data"""
        results = {}
        
        for lot_id in self.parking_lots.keys():
            # Filter data for this lot (simplified - in real implementation would match by coordinates)
            lot_data = current_data.iloc[0].to_dict()  # Simplified for demo
            
            if model_type == 'linear':
                new_price = self.model_1_baseline_linear(lot_data)
                rerouting = None
            elif model_type == 'demand_based':
                new_price = self.model_2_demand_based(lot_data)
                rerouting = None
            elif model_type == 'competitive':
                new_price, rerouting = self.model_3_competitive_pricing(lot_data, lot_id)
            
            # Update lot information
            self.parking_lots[lot_id]['current_price'] = new_price
            self.parking_lots[lot_id]['price_history'].append(new_price)
            
            results[lot_id] = {
                'price': new_price,
                'rerouting': rerouting,
                'occupancy_rate': lot_data['occupancy'] / lot_data['capacity'],
                'queue_length': lot_data['queue_length']
            }
        
        return results

class RealTimeSimulator:
    """
    Real-time simulation engine for parking data
    Simulates streaming data with timestamps
    """
    
    def __init__(self, data_path=None):
        self.data = self._generate_sample_data() if data_path is None else pd.read_csv(data_path)
        self.current_time_step = 0
        self.pricing_system = ParkingPricingSystem()
        
    def _generate_sample_data(self):
        """Generate sample parking data for demonstration"""
        np.random.seed(42)
        
        # Generate 14 parking lots over 73 days, 18 time points per day
        n_lots = 14
        n_days = 73
        n_timepoints = 18
        
        data = []
        
        # Generate unique locations
        base_lat, base_lon = 40.7128, -74.0060  # NYC coordinates
        
        for lot_id in range(n_lots):
            # Random location within 5km radius
            lat = base_lat + np.random.normal(0, 0.02)
            lon = base_lon + np.random.normal(0, 0.02)
            capacity = np.random.randint(20, 100)
            
            for day in range(n_days):
                for timepoint in range(n_timepoints):
                    # Generate realistic patterns
                    hour = 8 + timepoint * 0.5
                    
                    # Peak hours: 9-11 AM, 1-3 PM
                    peak_factor = 1.0
                    if (9 <= hour <= 11) or (13 <= hour <= 15):
                        peak_factor = 1.5
                    
                    # Weekend effect
                    is_weekend = day % 7 in [5, 6]
                    weekend_factor = 0.7 if is_weekend else 1.0
                    
                    # Random special day
                    is_special_day = np.random.random() < 0.1
                    
                    # Generate occupancy
                    base_occupancy = int(capacity * 0.3 * peak_factor * weekend_factor)
                    occupancy = max(0, min(capacity, base_occupancy + np.random.randint(-10, 15)))
                    
                    # Generate queue based on occupancy
                    queue_length = max(0, int((occupancy / capacity - 0.8) * 20)) if occupancy / capacity > 0.8 else 0
                    
                    data.append({
                        'day': day,
                        'timepoint': timepoint,
                        'hour': hour,
                        'latitude': lat,
                        'longitude': lon,
                        'capacity': capacity,
                        'occupancy': occupancy,
                        'queue_length': queue_length,
                        'vehicle_type': np.random.choice(['car', 'bike', 'truck'], p=[0.7, 0.2, 0.1]),
                        'traffic_level': np.random.randint(1, 11),
                        'is_special_day': is_special_day,
                        'lot_id': lot_id
                    })
        
        return pd.DataFrame(data)
    
    def simulate_real_time_stream(self, model_type='demand_based', n_steps=50):
        """Simulate real-time data stream and pricing updates"""
        
        # Initialize pricing system
        self.pricing_system.initialize_parking_lots(self.data)
        
        results_history = []
        
        print(f"Starting Real-Time Simulation with {model_type} model...")
        print("=" * 60)
        
        for step in range(n_steps):
            # Get current data batch
            current_data = self.data.iloc[step:step+14].copy()  # 14 lots
            
            if current_data.empty:
                break
            
            # Update prices
            pricing_results = self.pricing_system.update_prices(current_data, model_type)
            
            # Store results
            timestamp = datetime.now() + timedelta(minutes=step*30)
            
            for lot_id, result in pricing_results.items():
                results_history.append({
                    'timestamp': timestamp,
                    'lot_id': lot_id,
                    'price': result['price'],
                    'occupancy_rate': result['occupancy_rate'],
                    'queue_length': result['queue_length'],
                    'rerouting': result['rerouting']
                })
            
            # Print current status
            if step % 10 == 0:
                print(f"Step {step}: Average Price = ${np.mean([r['price'] for r in pricing_results.values()]):.2f}")
                
                # Show rerouting suggestions
                rerouting_suggestions = [r['rerouting'] for r in pricing_results.values() if r['rerouting']]
                if rerouting_suggestions:
                    print(f"  Rerouting suggestions: {len(rerouting_suggestions)} lots")
        
        return pd.DataFrame(results_history)

class VisualizationEngine:
    """
    Visualization engine for real-time pricing data
    """
    
    def __init__(self, results_df):
        self.results_df = results_df
        
    def plot_price_evolution(self, lot_ids=None):
        """Plot price evolution over time for selected lots"""
        if lot_ids is None:
            lot_ids = self.results_df['lot_id'].unique()[:4]  # Show first 4 lots
        
        plt.figure(figsize=(15, 8))
        
        for lot_id in lot_ids:
            lot_data = self.results_df[self.results_df['lot_id'] == lot_id]
            plt.plot(lot_data['timestamp'], lot_data['price'], label=f'{lot_id}', marker='o', markersize=3)
        
        plt.title('Real-Time Parking Price Evolution', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_occupancy_vs_price(self):
        """Plot relationship between occupancy and price"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.results_df['occupancy_rate'], self.results_df['price'], alpha=0.6)
        plt.xlabel('Occupancy Rate')
        plt.ylabel('Price ($)')
        plt.title('Price vs Occupancy Rate')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.results_df['queue_length'], self.results_df['price'], alpha=0.6, color='orange')
        plt.xlabel('Queue Length')
        plt.ylabel('Price ($)')
        plt.title('Price vs Queue Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results_dict):
        """Compare results from different models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(results_dict.keys())
        colors = ['blue', 'red', 'green']
        
        # Price comparison
        axes[0, 0].set_title('Average Price by Model')
        avg_prices = [results_dict[model]['price'].mean() for model in models]
        axes[0, 0].bar(models, avg_prices, color=colors[:len(models)])
        axes[0, 0].set_ylabel('Average Price ($)')
        
        # Price volatility
        axes[0, 1].set_title('Price Volatility by Model')
        price_std = [results_dict[model]['price'].std() for model in models]
        axes[0, 1].bar(models, price_std, color=colors[:len(models)])
        axes[0, 1].set_ylabel('Price Standard Deviation')
        
        # Price distribution
        axes[1, 0].set_title('Price Distribution')
        for i, model in enumerate(models):
            axes[1, 0].hist(results_dict[model]['price'], alpha=0.7, label=model, bins=20)
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Price ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Time series comparison
        axes[1, 1].set_title('Price Evolution Comparison')
        for model in models:
            model_data = results_dict[model]
            lot_data = model_data[model_data['lot_id'] == model_data['lot_id'].iloc[0]]
            axes[1, 1].plot(range(len(lot_data)), lot_data['price'], label=model, marker='o', markersize=3)
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Price ($)')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("Dynamic Parking Pricing System")
    print("=" * 50)
    
    # Initialize simulator
    simulator = RealTimeSimulator()
    
    # Run simulations with different models
    models = ['linear', 'demand_based', 'competitive']
    results = {}
    
    for model in models:
        print(f"\n{'='*20} {model.upper()} MODEL {'='*20}")
        results[model] = simulator.simulate_real_time_stream(model_type=model, n_steps=30)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Individual model visualization
    for model in models:
        print(f"\n{model.upper()} Model Results:")
        viz = VisualizationEngine(results[model])
        viz.plot_price_evolution()
        viz.plot_occupancy_vs_price()
    
    # Model comparison
    print("\nModel Comparison:")
    comparison_viz = VisualizationEngine(results['demand_based'])  # Use any for comparison
    comparison_viz.plot_model_comparison(results)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for model in models:
        data = results[model]
        print(f"\n{model.upper()} Model:")
        print(f"  Average Price: ${data['price'].mean():.2f}")
        print(f"  Price Range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        print(f"  Price Volatility: {data['price'].std():.2f}")
        print(f"  Avg Occupancy: {data['occupancy_rate'].mean():.2%}")
        
        # Count rerouting suggestions
        rerouting_count = data['rerouting'].notna().sum()
        print(f"  Rerouting Suggestions: {rerouting_count}")

if __name__ == "__main__":
    main()
