import matplotlib.pyplot as plt

reference_percentages = ['20%', '40%', '60%', '80%', '100%']
pass_at_1 = [45.37, 48.15, 55.55, 56.48, 56.48]

reference_percentages_reversed = reference_percentages[::-1]
pass_at_1_reversed = pass_at_1[::-1]

plt.figure(figsize=(7, 4))

plt.plot(reference_percentages_reversed, pass_at_1_reversed, marker='o', linestyle='-', linewidth=3, markersize=10, label='Pass@1', color='blue')

plt.xlabel("Reference Samples Percentage", fontsize=20)
plt.ylabel("Pass Rate (%)", fontsize=20)

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.tick_params(axis='both', which='major', labelsize=18, width=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("reference-percentage.pdf", bbox_inches='tight')
plt.show()
