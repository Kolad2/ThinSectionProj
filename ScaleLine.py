import matplotlib as mpl
import math

def plotScaleLine(ax, start, end):
	dx = (start[0] - end[0])
	dy = (start[1] - end[1])
	ca = dx / math.sqrt(dx * dx + dy * dy)
	sa = dy / math.sqrt(dx * dx + dy * dy)
	ax.plot([start[0], end[0]], [start[1], end[1]], transform=ax.transAxes, color='white', lw=5)
	ax.plot(
		[start[0] + 0.01 * sa, start[0] - 0.01 * sa],
		[start[1] - 0.01 * ca, start[1] + 0.01 * ca], transform=ax.transAxes, color='white', lw=5)
	ax.plot(
		[end[0] - 0.01 * sa, end[0] + 0.01 * sa],
		[end[1] + 0.01 * ca, end[1] - 0.01 * ca], transform=ax.transAxes, color='white', lw=5)
	#
	ax.plot([start[0], end[0]], [start[1], end[1]], transform=ax.transAxes, color='black', lw=2)
	ax.plot(
		[start[0] - 0.01 * sa, start[0] + 0.01 * sa],
		[start[1] + 0.01 * ca, start[1] - 0.01 * ca], transform=ax.transAxes, color='black', lw=2)
	ax.plot(
		[end[0] - 0.01 * sa, end[0] + 0.01 * sa],
		[end[1] + 0.01 * ca, end[1] - 0.01 * ca], transform=ax.transAxes, color='black', lw=2)