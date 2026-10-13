-- Schema definition for Executive Supply Chain & Sales Performance Database
-- Dialect: SQLite 3

DROP TABLE IF EXISTS orders;

CREATE TABLE orders (
    OrderID             TEXT PRIMARY KEY,
    OrderDate           TEXT NOT NULL,
    CustomerID          TEXT NOT NULL,
    CustomerSegment     TEXT NOT NULL CHECK(CustomerSegment IN ('Consumer', 'Corporate', 'Home Office')),
    Region              TEXT NOT NULL CHECK(Region IN ('North', 'South', 'East', 'West', 'Central')),
    Country             TEXT NOT NULL,
    City                TEXT NOT NULL,
    ProductCategory     TEXT NOT NULL CHECK(ProductCategory IN ('Electronics', 'Furniture', 'Office Supplies', 'Industrial')),
    SubCategory         TEXT NOT NULL,
    ProductSKU          TEXT NOT NULL,
    UnitPrice           REAL NOT NULL,
    Quantity            INTEGER NOT NULL CHECK(Quantity > 0),
    Discount            REAL NOT NULL CHECK(Discount >= 0.0 AND Discount <= 1.0),
    ShippingCost        REAL NOT NULL CHECK(ShippingCost >= 0.0),
    UnitCost            REAL NOT NULL CHECK(UnitCost >= 0.0),
    Sales               REAL NOT NULL,
    Profit              REAL NOT NULL,
    ProfitMargin        REAL NOT NULL,
    WarehouseID         TEXT NOT NULL CHECK(WarehouseID IN ('WH-North', 'WH-South', 'WH-East', 'WH-West', 'WH-Central')),
    ShippingMode        TEXT NOT NULL CHECK(ShippingMode IN ('Standard', 'Express', 'Same Day')),
    PromisedLeadDays    INTEGER NOT NULL CHECK(PromisedLeadDays >= 0),
    ActualLeadDays      INTEGER NOT NULL CHECK(ActualLeadDays >= 0),
    DeliveryStatus      TEXT NOT NULL CHECK(DeliveryStatus IN ('On-Time', 'Delayed', 'Cancelled')),
    InventoryLevel      INTEGER NOT NULL CHECK(InventoryLevel >= 0),
    StockoutRisk        TEXT NOT NULL CHECK(StockoutRisk IN ('Low', 'Medium', 'High'))
);

-- Performance Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(OrderDate);
CREATE INDEX IF NOT EXISTS idx_orders_region ON orders(Region);
CREATE INDEX IF NOT EXISTS idx_orders_category ON orders(ProductCategory);
CREATE INDEX IF NOT EXISTS idx_orders_delivery_status ON orders(DeliveryStatus);
CREATE INDEX IF NOT EXISTS idx_orders_warehouse ON orders(WarehouseID);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(CustomerID);
CREATE INDEX IF NOT EXISTS idx_orders_shipping_mode ON orders(ShippingMode);

-- Executive Monthly Aggregations View
DROP VIEW IF EXISTS v_monthly_executive_summary;
CREATE VIEW v_monthly_executive_summary AS
SELECT
    strftime('%Y-%m', OrderDate) AS YearMonth,
    strftime('%Y', OrderDate) AS Year,
    strftime('%m', OrderDate) AS Month,
    COUNT(OrderID) AS TotalOrders,
    COUNT(DISTINCT CustomerID) AS UniqueCustomers,
    SUM(Quantity) AS TotalUnitsSold,
    ROUND(SUM(Sales), 2) AS TotalRevenue,
    ROUND(SUM(Profit), 2) AS TotalProfit,
    ROUND(AVG(Sales), 2) AS AvgOrderValue,
    ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS ProfitMarginPct,
    ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OnTimeDeliveryRate,
    ROUND(SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS HighStockoutRate
FROM orders
GROUP BY strftime('%Y-%m', OrderDate);

-- Regional Supply Chain Bottleneck View
DROP VIEW IF EXISTS v_regional_fulfillment_performance;
CREATE VIEW v_regional_fulfillment_performance AS
SELECT
    Region,
    WarehouseID,
    ShippingMode,
    COUNT(OrderID) AS TotalShipments,
    ROUND(AVG(PromisedLeadDays), 1) AS AvgPromisedDays,
    ROUND(AVG(ActualLeadDays), 1) AS AvgActualDays,
    ROUND(AVG(CASE WHEN ActualLeadDays > PromisedLeadDays THEN ActualLeadDays - PromisedLeadDays ELSE 0 END), 2) AS AvgLeadDelayDays,
    ROUND(SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS DelayedRatePct,
    ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OnTimeRatePct,
    ROUND(SUM(ShippingCost), 2) AS TotalLogisticsCost,
    ROUND(SUM(ShippingCost) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS LogisticsCostRatioPct
FROM orders
GROUP BY Region, WarehouseID, ShippingMode;
