import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def optimizar_abastecimiento(demanda_predicha, c_orden, c_compra, c_inv, c_faltante, capacidad, inv_inicial=50):
    """
    Recibe la demanda de la IA y los parámetros financieros dinámicos desde la interfaz web.
    Retorna un DataFrame con el plan y el costo total.
    """
    T = len(demanda_predicha)
    periodos = range(T)
    V = 0.015     # Volumen en m3 por caja
    M = 100000    # Constante Big-M
    
    modelo = gp.Model("Optimizacion_Bavaria")
    modelo.setParam('OutputFlag', 0) 
    
    # Variables de Decisión
    Q = modelo.addVars(periodos, vtype=GRB.CONTINUOUS, name="Q")
    I = modelo.addVars(periodos, vtype=GRB.CONTINUOUS, name="I")
    F = modelo.addVars(periodos, vtype=GRB.CONTINUOUS, name="F")
    Y = modelo.addVars(periodos, vtype=GRB.BINARY, name="Y")
    
    # Función Objetivo
    modelo.setObjective(
        gp.quicksum(c_orden * Y[t] + c_compra * Q[t] + c_inv * I[t] + c_faltante * F[t] for t in periodos),
        GRB.MINIMIZE
    )
    
    # Restricciones
    for t in periodos:
        inv_anterior = inv_inicial if t == 0 else I[t-1]
        modelo.addConstr(I[t] == inv_anterior + Q[t] - demanda_predicha[t] + F[t])
        modelo.addConstr(Q[t] <= M * Y[t])
        modelo.addConstr(V * I[t] <= capacidad)
        
    modelo.optimize()
    
    if modelo.status == GRB.OPTIMAL:
        # Estructurar los resultados en una tabla limpia
        datos = []
        for t in periodos:
            datos.append({
                "Día": t + 1,
                "Demanda IA": int(demanda_predicha[t]),
                "Pedir (Q)": int(Q[t].X),
                "Inv. Final (I)": int(I[t].X),
                "Faltante (F)": int(F[t].X)
            })
        return pd.DataFrame(datos), modelo.objVal
    
    return None, None