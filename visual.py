import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self, title="Live Cumulative Rewards", xlabel="Episode", ylabel="Cumulative Reward"):
        # Initialize the plot
        self.cumulative_rewards = []
        self.episodes = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o', linestyle='-', color='b', label="Cumulative Reward")

        # Configure the plot
        self.ax.set_xlim(0, 10)  # Initial X-axis limits
        self.ax.set_ylim(-10, 50)  # Initial Y-axis limits
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid()
        self.ax.legend()
        plt.ion()  # Enable interactive mode

    def update(self, episode, cumulative_reward):
        # Append data
        self.cumulative_rewards.append(cumulative_reward)
        self.episodes.append(episode)

        # Update the plot data
        self.line.set_xdata(self.episodes)
        self.line.set_ydata(self.cumulative_rewards)

        # Dynamically adjust axes limits
        self.ax.set_xlim(0, len(self.episodes) + 1)
        self.ax.set_ylim(min(self.cumulative_rewards) - 5, max(self.cumulative_rewards) + 5)

        # Redraw the plot
        plt.pause(0.1)

    def show(self):
        # Finalize the plot
        plt.ioff()  # Disable interactive mode
        plt.show()
