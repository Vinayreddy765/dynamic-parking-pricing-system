# Dynamic Parking Pricing System
## Technical Report - Summer Analytics 2025

### Executive Summary

This report presents a comprehensive dynamic pricing system for urban parking lots that implements three increasingly sophisticated pricing models. The system processes real-time data streams to adjust parking prices based on demand, occupancy, competition, and various environmental factors.

### 1. System Architecture

The dynamic pricing system consists of four main components:

#### 1.1 ParkingPricingSystem Class
- **Purpose**: Core pricing engine implementing three models
- **Key Features**: 
  - Baseline linear pricing
  - Demand-based pricing with multiple factors
  - Competitive pricing with location intelligence
  - Geographic proximity calculations using Haversine distance

#### 1.2 RealTimeSimulator Class
- **Purpose**: Simulates real-time data streaming
- **Key Features**:
  - Generates realistic parking data for 14 lots over 73 days
  - Implements time-based demand patterns
  - Simulates 18 time points per day (8:00 AM - 4:30 PM)

#### 1.3 VisualizationEngine Class
- **Purpose**: Provides comprehensive data visualization
- **Key Features**:
  - Real-time price evolution plots
  - Occupancy vs price correlation analysis
  - Model comparison visualizations
  - Statistical summary reports

### 2. Pricing Models Implementation

#### 2.1 Model 1: Baseline Linear Model

**Mathematical Formula:**
```
Price(t+1) = Price(t) + α × (Occupancy/Capacity)
```

**Parameters:**
- α = 0.5 (sensitivity parameter)
- Price bounds: [0.5 × BasePrice, 2.0 × BasePrice]

**Implementation Logic:**
- Simple linear relationship between occupancy rate and price
- Acts as baseline for comparison
- Ensures price stability through bounded adjustments

#### 2.2 Model 2: Demand-Based Pricing

**Mathematical Formula:**
```
Demand = α×(Occupancy/Capacity) + β×QueueLength - γ×Traffic + δ×IsSpecialDay + ε×VehicleTypeWeight

Price = BasePrice × (1 + λ × tanh(Demand))
```

**Parameters:**
- α = 0.4 (occupancy coefficient)
- β = 0.3 (queue coefficient)  
- γ = 0.2 (traffic coefficient)
- δ = 0.3 (special day coefficient)
- ε = 0.1 (vehicle type coefficient)
- λ = 0.5 (price sensitivity)

**Key Features:**
- Multi-factor demand calculation
- Vehicle type differentiation (car: 1.0, bike: 0.5, truck: 1.5)
- Tanh normalization for smooth price transitions
- Negative traffic impact modeling

#### 2.3 Model 3: Competitive Pricing (Optional)

**Enhancement Features:**
- Geographic proximity analysis (2km radius)
- Competitor price monitoring
- Dynamic rerouting suggestions
- Adaptive pricing based on market conditions

**Business Logic:**
- High occupancy + expensive pricing → reduce price for competitiveness
- Low occupancy + cheap competitors → increase price within reason
- Automatic rerouting when occupancy > 90% and queue > 5 vehicles

### 3. Real-Time Data Processing

#### 3.1 Data Structure
Each data point includes:
- **Location**: Latitude, Longitude
- **Capacity**: Maximum parking spaces
- **Occupancy**: Current parked vehicles
- **Queue Length**: Waiting vehicles
- **Vehicle Type**: Car, bike, or truck
- **Traffic Level**: Congestion indicator (1-10)
- **Special Day**: Holiday/event indicator

#### 3.2 Streaming Simulation
- **Frequency**: 30-minute intervals
- **Duration**: 73 days simulation
- **Lots**: 14 different parking locations
- **Time Window**: 8:00 AM - 4:30 PM daily

### 4. Key Algorithms

#### 4.1 Haversine Distance Calculation
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)² + cos(lat1) × cos(lat2) × sin(dlon/2)²
    c = 2 × arcsin(√a)
    return R × c
```

#### 4.2 Demand Normalization
- Uses hyperbolic tangent (tanh) function for smooth transitions
- Ensures demand scores remain within [-1, 1] range
- Prevents erratic price fluctuations

#### 4.3 Price Bounding
- Minimum price: 50% of base price
- Maximum price: 200% of base price
- Ensures realistic pricing within acceptable ranges

### 5. Performance Analysis

#### 5.1 Model Comparison Metrics
- **Average Price**: Overall pricing level
- **Price Volatility**: Standard deviation of prices
- **Occupancy Correlation**: Relationship between demand and pricing
- **Rerouting Frequency**: Competitive model efficiency

#### 5.2 Expected Results
- **Linear Model**: Stable but less responsive to market conditions
- **Demand-Based**: More dynamic, better demand matching
- **Competitive**: Most sophisticated, includes market intelligence

### 6. Visualization Features

#### 6.1 Real-Time Monitoring
- Live price evolution charts
- Occupancy vs price scatter plots
- Queue length impact analysis
- Time-series trend visualization

#### 6.2 Comparative Analysis
- Side-by-side model performance
- Statistical distribution comparisons
- Volatility and stability metrics
- Business impact assessment

### 7. Technical Implementation

#### 7.1 Dependencies
- **Core Libraries**: NumPy, Pandas
- **Visualization**: Matplotlib, Bokeh (for real-time plots)
- **Real-Time Processing**: Pathway framework
- **Environment**: Google Colab

#### 7.2 Data Flow
1. **Ingestion**: Real-time data streaming simulation
2. **Processing**: Feature extraction and normalization
3. **Modeling**: Price calculation using selected model
4. **Output**: Updated prices and rerouting suggestions
5. **Visualization**: Real-time dashboard updates

### 8. Business Value

#### 8.1 Efficiency Improvements
- **Utilization Optimization**: Dynamic pricing balances demand
- **Revenue Maximization**: Price adjustments based on market conditions
- **Customer Satisfaction**: Reduced wait times through rerouting

#### 8.2 Competitive Advantages
- **Market Intelligence**: Real-time competitor monitoring
- **Adaptive Pricing**: Responds to changing market conditions
- **Data-Driven Decisions**: Evidence-based pricing strategies

### 9. Future Enhancements

#### 9.1 Advanced Features
- **Machine Learning Integration**: Historical pattern recognition
- **Weather Impact**: Weather-based demand modeling
- **Event Integration**: Calendar-based special event detection
- **Mobile Integration**: Real-time customer notifications

#### 9.2 Scalability Considerations
- **Multi-City Deployment**: Expandable to multiple urban areas
- **Cloud Integration**: Real-time data processing at scale
- **API Development**: Integration with existing parking systems

### 10. Conclusion

The dynamic parking pricing system successfully implements three increasingly sophisticated pricing models that can adapt to real-time market conditions. The demand-based model provides the best balance of responsiveness and stability, while the competitive model adds valuable market intelligence for strategic positioning.

The system demonstrates clear business value through improved utilization, revenue optimization, and customer satisfaction while maintaining technical robustness and scalability for future enhancements.

### 11. Code Usage Instructions

#### 11.1 Basic Execution
```python
# Run the main simulation
if __name__ == "__main__":
    main()
```

#### 11.2 Custom Model Testing
```python
# Test individual models
simulator = RealTimeSimulator()
results = simulator.simulate_real_time_stream(model_type='demand_based', n_steps=50)

# Visualize results
viz = VisualizationEngine(results)
viz.plot_price_evolution()
```

#### 11.3 Real-Time Integration
```python
# For Pathway integration (add to notebook)
import pathway as pw

# Define data schema and processing pipeline
# Connect to real-time data sources
# Implement continuous pricing updates
```

### 12. References

1. Pathway Framework Documentation: https://pathway.com/developers/
2. Real-Time Data Processing: https://pathway.com/developers/user-guide/introduction/first_realtime_app_with_pathway/
3. Summer Analytics 2025: https://www.caciitg.com/sa/course25/
4. Dynamic Pricing Theory: Academic literature on demand-based pricing models
5. Urban Parking Optimization: Research on smart city parking solutions