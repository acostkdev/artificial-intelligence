import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef

# ============================================================
# Ejercicio 1: Calcular metricas desde matriz de confusion
# ============================================================

# Matriz de confusion: [[VP, FP], [FN, VN]]
# Para un problema de deteccion de fraude (positivo = fraude)
VP = 45
FP = 10
FN = 15
VN = 30

total = VP + VN + FP + FN

accuracy = (VP + VN) / total
precision = VP / (VP + FP)
recall = VP / (VP + FN)
especificidad = VN / (VN + FP)
f1 = 2 * (precision * recall) / (precision + recall)

print("=== Ejercicio 1: Metricas desde matriz de confusion ===")
print(f"Matriz de confusion: VP={VP}, FP={FP}, FN={FN}, VN={VN}")
print(f"Total de muestras: {total}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
print(f"Especificidad: {especificidad:.4f} ({especificidad*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
print()

# ============================================================
# Ejercicio 2: Interpretar modelo con Accuracy 85%, Precision 40%, Recall 90%
# ============================================================

print("=== Ejercicio 2: Interpretacion de modelo ===")
acc_ej2 = 0.85
prec_ej2 = 0.40
rec_ej2 = 0.90

print(f"Accuracy={acc_ej2*100:.0f}%, Precision={prec_ej2*100:.0f}%, Recall={rec_ej2*100:.0f}%")
print("El modelo detecta bien los positivos (Recall alto) pero genera")
print("muchos falsos positivos (Precision baja). Si el juego requiere")
print("evitar falsas alarmas, no es buen modelo. Si priorizamos no")
print("perder ningun positivo, podria servir con ajustes.")
print()

# ============================================================
# Ejercicio 3: Proponer mejoras para modelo con Recall = 0.5
# ============================================================

print("=== Ejercicio 3: Mejoras para Recall bajo ===")
rec_ej3 = 0.5
print(f"Recall actual: {rec_ej3}")
print("Posibles mejoras:")
print("- Bajar el umbral de decision para clasificar mas positivos")
print("- Recolectar mas datos de entrenamiento")
print("- Balancear las clases si el dataset esta desbalanceado")
print("- Usar tecnicas de sobremuestreo como SMOTE")
print("- Probar con un modelo mas complejo o con regularizacion")
print("- Revisar si las caracteristicas usadas son relevantes")
print()

# ============================================================
# Curva ROC y AUC
# ============================================================

print("=== Curva ROC y AUC ===")

np.random.seed(42)
n_muestras = 200
y_real = np.random.randint(0, 2, n_muestras)
y_score = y_real + np.random.normal(0, 0.4, n_muestras)
y_score = np.clip(y_score, 0, 1)

fpr, tpr, thresholds_roc = roc_curve(y_real, y_score)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# ============================================================
# Curva Precision-Recall para clases desbalanceadas
# ============================================================

print("=== Curva Precision-Recall ===")

y_real_desb = np.zeros(500)
y_real_desb[:50] = 1

y_score_desb = y_real_desb + np.random.normal(0, 0.3, 500)
y_score_desb = np.clip(y_score_desb, 0, 1)

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_real_desb, y_score_desb)

plt.subplot(1, 3, 2)
plt.plot(recall_vals, precision_vals, color='blue', lw=2, label='Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall (10% positivos)')
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

# ============================================================
# Matthews Correlation Coefficient (MCC)
# ============================================================

print("=== Matthews Correlation Coefficient ===")

# Usamos los mismos valores del ejercicio 1
y_real_ej1 = np.array([1]*VP + [0]*VN + [1]*FN + [0]*FP)
y_pred_ej1 = np.array([1]*VP + [0]*VN + [0]*FN + [1]*FP)

mcc_val = matthews_corrcoef(y_real_ej1, y_pred_ej1)

print(f"MCC para matriz del ejercicio 1: {mcc_val:.4f}")

# Tambien probamos con un clasificador perfecto, uno aleatorio y uno inverso
y_perfecto = np.array([1, 1, 0, 0])
y_perfecto_pred = np.array([1, 1, 0, 0])
mcc_perfecto = matthews_corrcoef(y_perfecto, y_perfecto_pred)

y_aleatorio = np.array([1, 1, 0, 0])
y_aleatorio_pred = np.array([1, 0, 1, 0])
mcc_aleatorio = matthews_corrcoef(y_aleatorio, y_aleatorio_pred)

y_inverso = np.array([1, 1, 0, 0])
y_inverso_pred = np.array([0, 0, 1, 1])
mcc_inverso = matthews_corrcoef(y_inverso, y_inverso_pred)

print(f"MCC clasificador perfecto: {mcc_perfecto:.4f}")
print(f"MCC clasificador aleatorio: {mcc_aleatorio:.4f}")
print(f"MCC clasificador inverso: {mcc_inverso:.4f}")
print()

plt.subplot(1, 3, 3)
modelos = ['Perfecto', 'Aleatorio', 'Inverso', 'Ejercicio 1']
valores_mcc = [mcc_perfecto, mcc_aleatorio, mcc_inverso, mcc_val]
colores = ['green', 'orange', 'red', 'blue']
barras = plt.bar(modelos, valores_mcc, color=colores, alpha=0.7)
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
plt.ylabel('MCC')
plt.title('Comparacion de MCC')
plt.grid(axis='y', alpha=0.3)

for barra, val in zip(barras, valores_mcc):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(__file__.replace('.py', '_resultados.png'))
plt.show()

print("=== Resumen final ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Especificidad: {especificidad:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"MCC: {mcc_val:.4f}")
