"""论文式(1)、(2)与定理1。这里使用完整积欠，不重置负库存。"""
import numpy as np


def inventory_cost(inventory, h=1.0, b=9.0):
    """inventory必须是明确选定的逐期期末净库存。"""
    values = np.asarray(inventory, dtype=float)
    if h <= 0 or b <= 0 or not np.isfinite(values).all():
        raise ValueError("Positive h,b and finite inventories are required")
    holding = float(h * np.maximum(values, 0).sum())
    backorder = float(b * np.maximum(-values, 0).sum())
    return dict(holding_cost=holding, stockout_cost=backorder,
                total_cost=holding + backorder)


def simulate_backorders(demand, initial_stock, arrivals, h=1.0, b=9.0):
    """式(1)：到货在当日需求前；arrivals[t]为当日所有到货之和。"""
    demand, arrivals = np.asarray(demand, float), np.asarray(arrivals, float)
    if demand.ndim != 1 or arrivals.shape != demand.shape:
        raise ValueError("demand and arrivals must be equal-length one-dimensional arrays")
    if not np.isfinite(demand).all() or not np.isfinite(arrivals).all():
        raise ValueError("Non-finite demand or arrivals")
    if (demand < 0).any() or (arrivals < 0).any():
        raise ValueError("Demand and arrivals must be nonnegative")
    inventory = initial_stock + np.cumsum(arrivals - demand)
    return inventory, inventory_cost(inventory, h, b)


def hindsight_label(interval_demand, arrival_stock, h=1.0, b=9.0):
    """定理1：输入从本次到货到下次到货前的需求，返回非负订货量。

    index从0开始，故j0=floor(b*n/(h+b))，对应讲义中j*=j0+1。
    精确整数临界位置选右端，与正式版取整约定一致。
    """
    d = np.asarray(interval_demand, float)
    if d.ndim != 1 or len(d) == 0 or (d < 0).any() or not np.isfinite(d).all():
        raise ValueError("A non-empty nonnegative finite demand interval is required")
    if h <= 0 or b <= 0 or not np.isfinite(arrival_stock):
        raise ValueError("Invalid costs or arrival inventory")
    j0 = min(len(d) - 1, int(np.floor(b * len(d) / (h + b) + 1e-12)))
    target = float(d[:j0 + 1].sum())
    return dict(order=max(target - arrival_stock, 0.0), target=target,
                cutoff_index=j0, alpha=b / (h + b), interval_length=len(d))


def label_from_order_time(future_demand, initial_stock, lead, next_arrival, h=1.0, b=9.0):
    """没有其他在途到货时，将下单时库存转换为到货前库存。

    future_demand[0]是下单当天需求；本单在lead下标到达。
    next_arrival是下一单到货下标。两者仅供事后制标签，不能作在线输入。
    """
    if not 0 <= lead < next_arrival <= len(future_demand):
        raise ValueError("Invalid arrival boundaries")
    d = np.asarray(future_demand, float)
    arrival_stock = initial_stock - d[:lead].sum()
    result = hindsight_label(d[lead:next_arrival], arrival_stock, h, b)
    result['arrival_stock'] = float(arrival_stock)
    return result
