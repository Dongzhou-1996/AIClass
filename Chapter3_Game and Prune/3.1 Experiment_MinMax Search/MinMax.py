# 定义棋盘
class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]  # 3x3 棋盘
        self.current_player = 'X'  # 默认先手为 X
        self.human_player = None   # 人类玩家角色
        self.ai_player = None      # AI 玩家角色

    # 打印棋盘
    def print_board(self):
        print("-------------")
        for row in self.board:
            print("|", " | ".join(row), "|")
            print("-------------")

    # 检查是否有玩家获胜
    def check_winner(self, player):
        # 检查行
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        # 检查列
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        # 检查对角线
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    # 检查棋盘是否已满
    def is_board_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    # 获取所有可用的移动位置
    def get_available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    # 极大极小化算法
    def minimax(self, depth, is_maximizing):
        # 检查游戏是否结束
        if self.check_winner(self.ai_player):
            return 1
        if self.check_winner(self.human_player):
            return -1
        if self.is_board_full():
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = self.ai_player
                score = self.minimax(depth + 1, False)
                self.board[move[0]][move[1]] = ' '  # 撤销移动
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = self.human_player
                score = self.minimax(depth + 1, True)
                self.board[move[0]][move[1]] = ' '  # 撤销移动
                best_score = min(best_score, score)
            return best_score

    # AI 下棋
    def ai_move(self):
        best_score = -float('inf')
        best_move = None
        for move in self.get_available_moves():
            self.board[move[0]][move[1]] = self.ai_player
            score = self.minimax(0, False)
            self.board[move[0]][move[1]] = ' '  # 撤销移动
            if score > best_score:
                best_score = score
                best_move = move
        if best_move:
            self.board[best_move[0]][best_move[1]] = self.ai_player

    # 人类下棋
    def human_move(self):
        while True:
            try:
                row = int(input("请输入行号 (0, 1, 2): "))
                col = int(input("请输入列号 (0, 1, 2): "))
                if (row, col) in self.get_available_moves():
                    self.board[row][col] = self.human_player
                    break
                else:
                    print("该位置已被占用或无效，请重新输入！")
            except ValueError:
                print("请输入有效的数字！")

    # 开始游戏
    def play(self):
        # 选择角色
        self.human_player = input("请选择你的角色 (X 或 O): ").upper()
        while self.human_player not in ['X', 'O']:
            print("无效的选择！")
            self.human_player = input("请选择你的角色 (X 或 O): ").upper()
        self.ai_player = 'O' if self.human_player == 'X' else 'X'

        # 游戏循环
        while True:
            self.print_board()
            if self.check_winner(self.human_player):
                print("你赢了！")
                break
            if self.check_winner(self.ai_player):
                print("AI 赢了！")
                break
            if self.is_board_full():
                print("平局！")
                break

            if self.current_player == self.human_player:
                print("你的回合：")
                self.human_move()
            else:
                print("AI 的回合：")
                self.ai_move()

            # 切换玩家
            self.current_player = 'X' if self.current_player == 'O' else 'O'


# 运行游戏
if __name__ == "__main__":
    game = TicTacToe()
    game.play()
