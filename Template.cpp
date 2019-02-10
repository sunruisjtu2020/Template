// Template

// Data Structures:
// 并查集(带权，路径压缩，按秩合并)
struct uf_set {
	int f[10000];
	void reset() {
		for(int i = 1; i < 10000; ++i)
			f[i] = i;
	}
	int find(int x) { return f[x] = f[x] == x ? x : find(f[x]); }
	int merge(int a, int b) {
		int fa = find(a), fb = find(b);
		if(fa == fb) return 1;
		f[fa] = fb;
		return 0;
	}
};
// 手写堆
struct minroot_heap {
	int hp[10000], sum;
	void push(int x) {
		hp[++sum] = x;
		for(int i = sum, j = i >> 1; j; i = j, j = i >> 1)
			if(hp[i] < hp[j]) std::swap(hp[i], hp[j]);
	}
	void pop() {
		hp[1] = hp[sum--];
		for(int i = 1, j = i << 1; j <= sum; i = j, j = i << 1) {
			if(j + 1 <= sum && hp[j + 1] < hp[j]) ++j;
			if(hp[i] < hp[j]) break;
			else std::swap(hp[i], hp[j]);
		}
	}
};
// 左偏树
struct leftist_heap {
	int a[10000], s[10000][2], d[10000], f[10000];
	// a[]原始数据
	int merge(int u, int v) {
		if(!u && !v) return u + v;
		if(a[u] > a[v] || (a[u] == a[v] && u > v)) std::swap(u, v);
		s[u][1] = merge(s[u][1], v);
		f[s[u][1]] = u;
		if(d[s[u][0]] < d[s[u][1]]) std::swap(s[u][0], s[u][1]);
		d[u] = d[s[u][1]] + 1;
		return u;
	}
	int find(int x) { while(f[x]) x = f[x]; return x; }
	void merge_node(int u, int v) {
		int fu = find(u), fv = find(v);
		merge(fu, fv);
	}
	int kill_node(int x) {
		if(a[x] < 0) return 1;
		int fx = find(x);
		int ans = a[fx];
		a[fx] = -1;
		f[s[fx][0]] = f[s[fx][1]] = 0;
		merge(s[fx][0], s[fx][1]);
		return ans;
	}
}
// 树状数组
struct binTree {
	int tr[10000];
	void add(int x, int v) {
		for(; x < 10000; x += (x & -x))
			tr[x] += v;
	}
	int sum(int x) {
		int ret = 0;
		for(; x > 0; x -= (x & -x))
			ret += tr[x];
		return ret;
	}
};
// 线段树
// STL(set, map, sort, next_permutation, priority_queue, unique)
#include <bitset>//bitset 需要的头文件
using namespace std;
int maxn = 10000;
bitset <maxn> p; // bitset <大小> 名称
//常用： 
p[pos] //取出第 pos 位
p.reset() //全部置零
p.set() //全部置一
p.count() //获取一的个数
p.any() //是否有一 
p.none() //是否全为零 
p.flip() //全部取反
p.flip(k) //第 k 位取反
p.to_ulong() //返回 p 转换成 unsigned long 的结果，超范围报错
p.to_ullong() //返回 p 转换成 unsigned long long 的结果，超范围报错
bitset <100> a, b, c;
a = b | c;
a = b & c;
a = b ^ c;
int a;
bitset <maxn> p
p = p & a;
p = p | a;
p = p ^ a;
//复杂度:O(n/32)
//bitset也同样支持左移右移 
a = a << 10;
b = b >> 10; 
// ST表

//DP
/* Dilworth定理
求一个序列里面最少有多少最长不上升序列等于求这个序列里最长上升序列的长度。
*/
// 最长不升子序列
/* 导弹拦截
一发炮弹都不能高于前一发的高度
只有一套系统
计算这套系统最多能拦截多少导弹，
如果要拦截所有导弹最少要配备多少套这种导弹拦截系统。
*/
const int M = 1e5 + 5;
int a[M], f[M] = {23333333}, n, l, r, mid, dan, lan;
int main() {
	while(~scanf("%d", &a[++n]));
	--n;
	for(int i = 1; i <= n; ++i) {
		if(f[dan] >= a[i]) f[++dan] = a[i];
		else {
			l = 0, r = dan;
			while(l <= r) {
				mid = l + r >> 1;
				if(f[mid] >= a[i]) l = mid + 1;
				else r = mid - 1;
			}
			if(l) f[l] = a[i];
		}
	}
	printf("%d\n", dan);
	memset(f, -1, sizeof(f));
	for(int i = 1; i <= n; ++i) {
		if(f[lan] < a[i]) f[++lan] = a[i];
		else {
			l = 0, r = lan;
			while(l <= r) {
				mid = l + r >> 1;
				if(f[mid] >= a[i]) r = mid - 1;
				else l = mid + 1;
			}
			f[l] = a[i];
		}
	}
	printf("%d\n", lan);
}
// 最长公共子序列
/*
给出1-n的两个排列P1和P2，求它们的最长公共子序列。
*/
int p1, p2[M], f[M], g[M], ans, n;
void dp() {
	scanf("%d", &n);
	for(int i = 1; i <= n; ++i)
		scanf("%d", &p1), g[p1] = i;
	for(int i = 1; i <= n; ++i)
		scanf("%d", p2 + i);
	memset(f, 0x7f, sizeof(f));
	f[0] = 0;
	for(int i = 1; i <= n; ++i) {
		if(g[p2[i]] > f[ans]) f[++ans] = g[p2[i]];
		else {
			int l = 1, r = n, mid = l + r >> 1;
			for(; l < r; mid = l + r >> 1) {
				if(g[p2[i]] < f[mid]) r = mid;
				else l = mid + 1;
			}
			f[l] = f[l] < g[p2[i]] ? f[l] : g[p2[i]];
		}
	}
	printf("%d\n", ans);
}
//背包九讲(分组，多重，二维)
//数位
long long dp[20][];
long long dfs(int pos, bool lead, bool limit) {
	// pos 状态变量 lead前导0 limit 数位上界变量
	if(pos == -1) return 1; // 递归基
	if(!limit && !lead && dp[pos][state] != -1) return dp[pos][state];
	int up = limit ? a[pos] : 9;
	long long ans = 0;
	for(int i = 0; i <= up; ++i) {
		if() //...
		else if() //...
		ans += dfs(pos - 1, lead && i == 0, limit && i == a[pos]);
	}
	if(!limit && !lead) dp[pos][state] = ans;
	return ans;
}
/*
windy定义了一种windy数。
不含前导零且相邻两个数字之差至少为2的正整数被称为windy数。 
windy想知道，在A和B之间，包括A和B，总共有多少个windy数？
*/
int a[20], dp[20][20], lef, rig;
int dfs(int pos, int last, int lim, int lead) {
	if(pos == 0) return 1;
	if(!lead && !lim && dp[pos][last] != -1)
		return dp[pos][last];
	int p, cnt = 0, up = lim ? a[pos] : 9;
	for(int i = 0; i <= up; ++i) {
		if(abs(i - last) < 2) continue;
		p = i;
		if(lead && i == 0) p = -19260817;
		cnt += dfs(pos - 1, p, lim && i == up, (p == -19260817));
	}
	if(!lim && !lead) dp[pos][last] = cnt;
	return cnt;
}
int solve(int x) {
	int t = 0;
	for(; x; x /= 10)
		a[++t] = x % 10;
	memset(dp, -1, sizeof(dp));
	return dfs(t, -19260817, 1, 1);
}
int main() {
	int aa, bb;
	scanf("%d%d", &aa, &bb);
	printf("%d\n", solve(bb) - solve(aa - 1));
}
//区间
	// NOI1995 石子合并
/*
在一个圆形操场的四周摆放N堆石子,现要将石子有次序地合并成一堆.
规定每次只能选相邻的2堆合并成新的一堆，
并将新的一堆的石子数，记为该次合并的得分。
试设计出1个算法,计算出将N堆石子合并成1堆的最小得分和最大得分.
*/
int n, maxc, minc = 1e9, f[305][305], g[305][305], a[305], s[305];
void dp() {
	scanf("%d", &n);
	for(int i = 1; i <= n; ++i)
		scanf("%d", &a[i]);
	for(int i = 1; i <= n; ++i)
		a[i + n] = a[i];
	for(int i = 1; i <= 2 * n; ++i)
		s[i] = s[i - 1] + a[i];
	for(int b = 1; b < n; ++b)
		for(int i = 1, j = i + b; j <= 2 * n && i <= 2 * n; ++i, j = i + b) {
			g[i][j] = 1e9;
			for(int k = i; k < j; ++k) {
				f[i][j] = std::max(f[i][j], f[i][k] + f[k + 1][j] - s[i - 1] + s[j]);
				g[i][j] = std::min(g[i][j], g[i][k] + g[k + 1][j] - s[i - 1] + s[j]);
			}
		}
	for(int i = 1; i <= n; ++i) {
		maxc = std::max(maxc, f[i][i + n - 1]);
		minc = std::min(minc, g[i][i + n - 1]);
	}
	printf("%d\n%d\n", minc, maxc);
}
//状压
//单调队列优化
/*
NOI2005瑰丽华尔兹
1个 n × m 的矩形网格。你初始站在(x, y)这个格。
有些格有障碍有些没有。
有 K 个时间段。第个时间段从si持续到ti（包括两端，即其长t−s+1个时间单位）。
这段时间内网格会向某个方向（上下左右之一）倾斜。
所以每个时间段内的每个时间单位，你可以选择在原地不动，
或者向倾斜的方向走格（当然你不能⾛到障碍上或是走出网格）。
求你最多能走多少格。
*/
#include <bits/stdc++.h>
const int M = 205;
int n, m, zx, zy, k, s, t, d, len, ans, f[M][M], dl[M], l, r, dp[M], p[M];
int tx[5] = {0, -1, 1, 0, 0};
int ty[5] = {0, 0, 0, -1, 1};
char a[M][M];
inline int jud(int x, int y) { return x > 0 && x <= n && y > 0 && y <= m; } // border
inline void getAns(int x, int y) {
	l = 1, r = 0; // clear queue
	for(int i = 1; jud(x, y); ++i, x += tx[d], y += ty[d]) { // find ans
		if(a[x][y] == 'x') l = 1, r = 0; // can't move, clear queue
		else if(a[x][y] == '.') {
			while(l <= r && dp[dl[r]] + i - p[dl[r]] < f[x][y]) --r; // not empty and now status > dp[r]
			dl[++r] = i, dp[i] = f[x][y], p[i] = i; // push now status
			while(p[dl[r]] - p[dl[l]] > len) ++l; // time limit
			f[x][y] = dp[dl[l]] + i - p[dl[l]]; // now status = now best status + move length
			ans = std::max(ans, f[x][y]); // update ans
		}
	}
}
int main() {
	scanf("%d%d%d%d%d", &n, &m, &zx, &zy, &k);
	for(int i = 1; i <= n; ++i)
		for(int j = 1; j <= m; ++j)
			std::cin >> a[i][j];
	memset(f, 0x80, sizeof(f)), f[zx][zy] = 0; // set array
	for(int i = 1; i <= k; ++i) {
		scanf("%d%d%d", &s, &t, &d);
		len = t - s + 1;
		if(d == 1) for(int j = 1; j <= m; ++j) getAns(n, j); // up
		if(d == 2) for(int j = 1; j <= m; ++j) getAns(1, j); // down
		if(d == 3) for(int j = 1; j <= n; ++j) getAns(j, m); // left
		if(d == 4) for(int j = 1; j <= n; ++j) getAns(j, 1); // right
	}
	printf("%d\n", ans);
}
//斜率优化
// 图论
// Floyed
void floyed() {
	for(int k = 1; k <= n; ++k)
		for(int i = 1; i <= n; ++i)
			for(int j = 1; j <= n; ++j)
				dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k][j]);
}
// Dijkstra
struct edge { edge* nxt = NULL; int to, w; }e[M << 1], *h[M];
struct node { int w, id; };
int operator <(node a, node b) { return a.w < b.w; }
namespace heap {
	node hp[M << 1];
	int sum;
	inline void push(node x) {
		hp[++sum] = x;
		for(int i = sum, j = i >> 1; j; i = j, j = i >> 1)
			if(hp[i] < hp[j]) std::swap(hp[i], hp[j]);
	}
	inline void pop() {
		hp[1] = hp[sum--];
		for(int i = 1, j = i << 1; j <= sum; i = j, j = i << 1) {
			if(j + 1 <= sum && hp[j + 1] < hp[j]) ++j;
			if(hp[i] < hp[j]) break;
			else std::swap(hp[i], hp[j]);
		}
	}
	inline node top() { return hp[1]; }
	inline int size() { return sum; }
}
inline void add_edge(int u, int v, int w) {
	e[++tot].nxt = h[u], h[u] = e + tot, h[u] -> to = v, h[u] -> w = w;
}
inline void spfa() { // dijkstra
	heap::push((node) {0, s});
	memset(d, 0x7f, sizeof(d));
	d[s] = 0;
	while(heap::size()) {
		node now = heap::top();
		heap::pop();
		register int x = now.id;
		if(now.w ^ d[x]) continue;
		for(edge* i = h[x]; i != NULL; i = i -> nxt)
			if(d[i -> to] > d[x] + i -> w) {
				d[i -> to] = d[x] + i -> w;
				heap::push((node) {d[i -> to], i -> to});
			}
	}
}
// SPFA
void spfa(int s) {
	std::queue<int> q;
	memset(d, 0x7f, sizeof(d));
	q.push(s), inq[s] = 1;
	while(q.size()) {
		int x = q.front();
		q.pop(); inq[x] = 0;
		for(int i = h[x]; i; i = nxt[i]) {
			if(d[to[i]] > d[x] + w[i]) {
				d[to[i]] = d[x] + w[i];
				if(!inq[to[i]]) q.push(to[i]), inq[to[i]] = 1;
			}
		}
	}
}
// MST Kruskal
void Kruskal() {
	for(int i = 1; i <= n; ++i)
		f[i] = i;
	int k = 0, ans = 0;
	std::sort(e + 1, e + m + 1);
	for(int i = 1; i <= m; ++i) {
		int fx = find(e[i].x), fy = find(e[i].y);
		if(fx != fy) f[fx] = fy, ++k, ans += e[i].w;
		if(k == n - 1) break;
	}
}
// MST Prim
struct ps { int id, wi; };
int operator <(ps a, ps b) {
	return a.wi > b.wi;
}
std::priority_queue<ps> q;
void prim() {
	memset(dis, 0x7f, sizeof(dis));
	dis[1] = 0;
	q.push((ps) {1, 0});
	while(q.size() && cnt < n) {
		int wi = q.top().wi, x = q.top().id;
		q.pop();
		if(inq[x]) continue;
		++cnt;
		sum += wi;
		inq[x] = 1;
		for(int i = h[x]; i; i = nxt[i])
			if(w[i] < dis[to[i]])
				dis[to[i]] = w[i], q.push((ps) {to[i], w[i]});
	}
}
// 拓扑排序
void TopSort() {
	for(int i = 1; i <= n; ++i)
		if(ind[i] == 0) st[++top] = i;
	while(top) {
		int x = st[top--];
		dfn[++cnt] = x;
		for(int i = h[x]; i; i = nxt[i])
			if(--ind[to[i]] == 0) st[++top] = to[i];
	}
}
// 倍增LCA
void dfs(int x) {
	for(int i = 1; i <= 19; ++i)
		f[x][i] = f[f[x][i - 1]][i - 1];
	for(int i = h[x]; i; i = nxt[i])
		if(!d[to[i]]) {
			d[to[i]] = d[x] + 1;
			f[to[i]][0] = x;
			dfs(to[i]);
		}
}
int lca(int u, int v) {
	if(u == v) return u;
	if(d[u] < d[v]) std::swap(u, v);
	for(int i = 19; i >= 0; --i)
		if(d[v] <= d[f[u][i]]) u = f[u][i];
	if(u == v) return u;
	for(int i = 19; i >= 0; --i)
		if(f[u][i] != f[v][i])
			u = f[u][i], v = f[v][i];
	return f[u][0];
}
void work(int a, int b) {
	d[1] = 1;
	dfs(1);
	int lcaa = lca(a, b);
}
// Tarjan缩点
void dfs(int x) {
	dfn[x] = low[x] = ++dfn[0];
	vis[x] = 1, st[++top] = x;
	for(int i = h[x]; i; i = nxt[i]) {
		if(!dfn[to[i]]) dfs(to[i]), low[x] = std::min(low[x], low[to[i]]);
		else if(vis[to[i]]) low[x] = std::min(low[x], dfn[to[i]]);
	}
	if(dfn[x] == low[x]) {
		++scc[0];
		while(st[top + 1] != x) {
			scc[st[top]] = scc[0];
			vis[st[top]] = 0;
			--top;
		}
	}
}
// 割点
void dfs(int x) {
	dfn[x] = low[x] = ++dfn[0];
	int rec = 0;
	for(int i = h[x]; i; i = nxt[i]) {
		if(!dfn[to[i]]) {
			f[to[i]] = f[x];
			dfs(to[i]);
			low[x] = std::min(low[x], low[to[i]]);
			if(low[to[i]] >= dfn[x] && x != f[x]) fuc[x] = 1; // 标记割点
			if(x == fuc[x]) ++rec;
		}
		low[x] = std::min(low[x], dfn[to[i]]);
	}
	if(x == f[x] && rec >= 2) fuc[f[x]] = 1; // 标记割点
}
// 次短路
// MaxFlow
int bfs(int s, int t) {
	memset(d, 0, sizeof(d));
	std::queue<int> q;
	q.push(s);
	d[s] = 1;
	for(int i = 1; i <= t; ++i)
		cur[i] = h[i];
	while(q.size()) {
		int v = q.front();
		q.pop();
		for(int i = h[v]; i; i = nxt[i])
			if(w[i] > 0 && !d[to[i]])
				d[to[i]] = d[v] + 1, q.push(to[i]);
	}
	return d[t];
}
long long dfs(int v, int t, long long flow) {
	if(v == t) return flow;
	long long k = 0;
	for(int i = cur[v]; i && k < flow; i = nxt[i]) {
		cur[v] = i;
		if(w[i] > 0 && d[to[i]] == d[v] + 1) {
			long long spfa = dfs(to[i], t, std::min(w[i], flow - k));
			if(spfa > 0) w[i] -= spfa, w[i ^ 1] += spfa, k += spfa;
			else d[to[i]] = -1;
		}
	}	
	return k;
}
long long dinic(int s, int t) {
	while(bfs(s, t)) maxFlow += dfs(s, t);
	return maxFlow;
}
// 二分图染色
bool dfs(int x) {
	vis[x] = 1;
	for(int i = h[x]; i; i = nxt[i]) {
		if(vis[to[i]]) {
			if(col[x] == col[to[i]]) return 1; // 不是二分图
		} else {
			col[to[i]] = col[x] ^ 1;
			if(dfs(to[i])) return 1; // 不是二分图
		}
	}
	return 0; // 是二分图
}
// 差分约束
	// spfa判负环
int spfa(int x) {
	vis[x] = 1;
	for(edge* i = h[x]; i != NULL; i = i -> nxt) {
		if(d[i -> to] < d[x] + i -> w) {
			d[i -> to] = d[x] + i -> w;
			if(vis[i -> to]) return 0;
			if(!spfa(i -> to)) return 0;
		}
	}
	vis[x] = 0;
	return 1;
}
// 负环
int spfa() {
	std::queue<int> q;
	d[1] = 0, vis[1] = 1;
	q.push(1);
	while(q.size()) {
		int p = q.front();
		q.pop();
		vis[p] = 0;
		for(int i = h[p]; i; i = nxt[i])
			if(d[p] + w[i] < d[to[i]]) {
				d[to[i]] = d[p] + w[i];
				if(!vis[to[i]]) {
					num[to[i]] = num[p] + 1;
					if(num[to[i]] >= n) return 1;
					q.push(to[i]), vis[to[i]] = 1;
				}
			}
	}
	return 0;
}
void spfa(int s) {
	if(flg) return;
	vis[s]=1;
	for(int i=h[s];i;i=e[i].nxt) {
		if(flg) return;
		if(d[s]+e[i].w<d[e[i].to]) {
			d[e[i].to]=d[s]+e[i].w;
			if(vis[e[i].to]) {
				flg=1; return;
			}
			else spfa(e[i].to);
		}
	}
	vis[s]=0;
}
// 2-SAT
/*
有n个布尔变量x1~xn，另有m个需要满足的条件，每个条件的形式都是“xi
为true/false或xj为true/false”。比如“x1为真或x3为假”、“x7为假或x2为假”。
2-SAT 问题的目标是给每个变量赋值使得所有条件得到满足。
第一行两个整数n和m，意义如体面所述。接下来m行每行4个整数 i a j b，
表示“xi为a或xj为b”(a,b∈{0,1})
如无解，输出“IMPOSSIBLE”（不带引号）;否则输出"POSSIBLE"（不带引号),
下 一行n个整数x1~xn xi∈{0,1}）表示构造出的解。
*/
const int M = 2e6 + 5;
struct edge { edge* nxt; int to; }e[M << 1], *h[M];
int n, m, tot, top, st[M], low[M], dfn[M], scc[M], vis[M], ans[M], vpn[M];
inline void add_edge(int u, int v) {
	e[++tot].nxt = h[u];
	h[u] = e + tot;
	h[u] -> to = v;
}
void dfs(int x) {
	dfn[x] = low[x] = ++dfn[0];
	vis[x] = vpn[x] = 1;
	st[++top] = x;
	for(edge* i = h[x]; i != NULL; i = i -> nxt) {
		if(!dfn[i -> to]) dfs(i -> to), low[x] = std::min(low[x], low[i -> to]);
		else if(vis[i -> to]) low[x] = std::min(low[x], dfn[i -> to]);
	}
	if(low[x] == dfn[x]) {
		++scc[0];
		while(top) {
			scc[st[top]] = scc[0];
			vis[st[top]] = 0;
			if(st[top--] == x) break;
		}
	}
}
int main() {
	scanf("%d%d", &n, &m);
	for(int i = 1, x, y, z, w; i <= m; ++i) {
		scanf("%d%d%d%d", &x, &y, &z, &w);
		add_edge(x + (y ? 0 : n), z + (w ? n : 0));
		add_edge(z + (w ? 0 : n), x + (y ? n : 0));
	}
	for(int i = 1; i <= n * 2; ++i, dfn[0] = 0)
		if(!vpn[i]) dfs(i);
	for(int i = 1; i <= n; ++i) {
		if(scc[i] == scc[i + n]) {
			puts("IMPOSSIBLE");
			return 0;
		}
		if(scc[i] > scc[i + n]) ans[i] = 1;
	}
	puts("POSSIBLE");
	for(int i = 1; i <= n; ++i)
		printf("%d ", ans[i]);
	return 0;
}
// 树链剖分
void dfs(int x) {
	siz[x] = 1;
	for(int i = h[x]; i; i = nxt[i]) {
		if(d[to[i]]) continue;
		d[to[i]] = d[x] + 1;
		f[to[i]] = x;
		dfs(to[i]);
		siz[x] += siz[to[i]];
		if(siz[son[x]] < siz[to[i]]) son[x] = to[i];
	}
}
void sfd(int x, int tp) {
	top[x] = tp;
	dfn[x] = ++dfn[0], pre[dfn[0]] = x;
	if(son[x]) sfd(son[x], tp);
	for(int i = h[x]; i; i = nxt[i])
		if(to[i] != f[x] && to[i] != son[x])
			sfd(to[i], to[i]);
}

// 数学
// gcd & exgcd
int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
void exgcd(int a, int b, int& x, int& y) {
	if(b == 0) x = 1, y = 0;
	else exgcd(b, a % b, y, x), y -= a / b * x;
}
// 筛法
void get_prime(int maxc) {
	vis[1] = 1;
	for(int i = 2; i <= maxc; ++i) {
		if(!vis[i]) pr[++tot] = i;
		for(int j = 1; j <= tot && i * pr[j] <= maxc; ++j) {
			vis[i * pr[j]] = 1;
			if(i % pr[j] == 0) break;
		}
	}
}
// 欧拉函数
void get_phi(int maxc) {
	phi[1] = 1;
	for(int i = 2; i <= maxc; ++i) {
		if(!v[i]) pr[++tot] = i, phi[i] = i - 1;
		for(int j = 1; j <= tot && i * pr[j] <= maxc; ++j) {
			v[i * pr[j]] = 1;
			if(i % pr[j] == 0) { phi[i * pr[j]] = phi[i] * pr[j]; break; }
			else phi[i * pr[j]] = phi[i] * (pr[j] - 1);
		}
	}
}
int phi(int x) {
	int ans = n;
	for(int i = 2; i * i <= n; ++i)
		if(x % i == 0) {
			ans = ans / i * (i - 1);
			while(x % i == 0) x /= i;
		}
	if(x > 1) ans = ans / x * (x - 1);
	return ans;
}
// 逆元
int inv(int a, int m) {
	// a^{-1} (mod m)
	if(a == 0) return 0;
	int ta = a, tb = m, x = 0, y = 0;
	exgcd(ta, tb, x, y);
	x = (x % tb + tb) % tb;
	if(x == 0) x = tb;
	return x;
}
qpow(a, m - 2, m); // a^{-1} = a^{\phi(p)-1}
inv[1] = 1;
for(int i = 2; i <= n; ++i)
	inv[i] = ((-(p / i) * inv[p % i]) % p + p) % p;
// Lucas定理
int comb(int n, int m, int p) {
	// c(n,m)%p
	// inv[i] 是 i! (mod p)的逆元 fac[i] = i
	if(m > n) return 0;
	if(n < p && m < p) return fac[n] * inv[m] % p * inv[n - m] % p;
	return comb(n / p, m / p, p) * comb(n % p, m % p, p) % p;
}
// inv[n!]递推
fac[0] = fac[1] = inv[0] = inv[1] = 1;
for(int i = 2; i < p; ++i)
	fac[i] = fac[i - 1] * i % p;
for(int i = 2; i < p; ++i)
	inv[i] = (p - p / i) * inv[p % i] % p;
for(int i = 1; i < p; ++i)
	inv[i] = inv[i - 1] * inv[i] % p;
// 矩阵加速
struct mat {
	int a[3][3];
	mat() { memset(a, 0, sizeof(a)); }
};
mat operator *(mat a, mat b) {
	mat c;
	memset(c.a, 0, sizeof(c.v));
	for(int k = 0; k < 3; ++k)
		for(int i = 0; i < 3; ++i)
			for(int j = 0; j < 3; ++j)
				c.a[i][j] += a.a[i][k] * b.a[k][j];
	return mat;
}
// 莫比乌斯反演
// \sum_{i=1}^{n}{\sum_{j=1}^{m}{[\gcd(i, j) = prime]}}
void get_mu(int maxc) {
	mu[1] = 1;
	for(int i = 2; i <= maxc; ++i) {
		if(!v[i]) mu[i] = -1, pr[++tot] = i;
		for(int j = 1; j <= tot && i * pr[j] <= maxc; ++j) {
			v[i * pr[j]] = 1;
			if(i % pr[j] == 0) { mu[i * pr[j]] = 0; break; }
			else mu[i * pr[j]] = -mu[i];
		}
	}
	for(int j = 1; j <= tot; ++j)
		for(int i = 1; i * pr[j] <= maxc; ++i)
			g[i * pr[j]] += mu[i];
	for(int i = 1; i <= maxc; ++i)
		g[i] += g[i - 1];
}
void mobius_inver(int n, int m) {
	if(n > m) std::swap(n, m);
	int ans = 0;
	for(int l = 1, r; l <= n; l = r + 1) {
		r = std::min(n / (n / l), m / (m / l));
		ans += 1ll * (n / l) * (m / l) * (g[r] - g[l - 1]);
	}
	printf("%d\n", ans);
}
// \sum_{i=a}^{b}{\sum{j=c}^{d}{[\gcd(i,j)=k]}}
void get_mu(int maxc) {
	mu[1] = 1;
	for(int i = 2; i <= maxc; ++i) {
		if(!v[i]) mu[i] = -1, pr[++tot] = i;
		for(int j = 1; j <= tot && i * pr[j] <= maxc; ++j) {
			v[i * pr[j]] = 1;
			if(i % pr[j] == 0) { mu[i * pr[j]] = 0; break; }
			else mu[i * pr[j]] = -mu[i];
		}
	}
	for(int i = 1; i <= maxc; ++i)
		g[i] = g[i - 1] + mu[i];
}
int mobius_inver(int n, int m) {
	if(n > m) std::swap(n, m);
	int ans = 0;
	for(int l = 1, r; l <= n; l = r + 1) {
		r = std::min(n / (n / l), m / (m / l));
		ans += (n / l) * (m / l) * (g[r] - g[l - 1]);
	}
	return ans;
}
void get_ans(int a, int b, int c, int d, int k) {
	--a, --c;
	a /= k, b /= k, c /= k, d /= k;
	int ans = mobius_inver(a, c) + mobius_inver(b, d);
	ans -= mobius_inver(a, d) - mobius_inver(b, c);
	printf("%d\n", ans);
}
// 卡特兰数递推
f[0] = f[1] = 1;
f[n] = f[0] * f[n - 1] + f[1] * f[n - 2] + ... + f[n - 1] * f[0]
f[n] = c(2n, n) / (n + 1)
f[n] = f[n - 1] * (4 * n - 2) / (n + 1);

// 字符串
// Hash
const int M = 19260823, BS = 31;
int has[], pw[] = {1};
void prework(char* s) {
	has[0] = (s[0] + M) % M;
	for(int i = 1; s[i]; ++i)
		has[i] = ((has[i - 1] * BS) % M + s[i] + M) % M;
	for(int i = 1; i <= 23333; ++i)
		pw[i] = 1ll * pw[i - 1] * BS % M;
}
int hash(int l, int r) {
	if(l == 0) return has[r];
	return ((has[r] - has[l - 1] * pw[r - l + 1] % M) % M + M) % M;
}
// KMP
void preNxt(char* s) {
	int ls = strlen(s);
	int j = 0;
	for(int i = 1; i < ls; ++i) {
		while(j && s[i] != s[j]) j = nxt[j];
		if(s[i] == s[j]) ++j;
		nxt[i + 1] = j;
	}
}
// Trie树
int tot, trie[2333][26];
int isw[2333], cnt[2333], usd[2333];
void insert(char* s, int rt) {
	for(int i = 0; s[i]; ++i) {
		int x = s[i] - 'a';
		if(trie[rt][x] == 0)
			trie[rt][x] = ++tot;
		rt = trie[rt][x];
	}
	isw[rt] = 1;
	++cnt[rt];
}
int find(char* s, int rt) {
	for(int i = 0; s[i]; ++i) {
		int x = s[i] - 'a';
		if(trie[rt][x] == 0) return 0;
		rt = trie[rt][x];
	}
	int flag = usd[rt];
	usd[rt] = 1; // 是否曾经被查询过
	return flag ? 2 : isw[rt];
}
// 最小表示法
int getMin(char* s) {
	int i = 0, j = 1, k = 0, len = strlen(s);
	while(i < len && j < len && k < len) {
		int cmp = s[(i + k) % len] - s[(j + k) % len];
		if(cmp) {
			cmp > 0 ? i += k + 1 : j += k + 1;
			if(i == j) ++j;
			k = 0;
		} else ++k;
	}
	return std::min(i, j);
}

// 乱搞
// 模拟退火
	// jsoi2004平衡点
void get_energy(double sx, double sy) {
	double ret = 0;
	for(int i = 1; i <= n; ++i) {
		double dx = sx - a[i].x, dy = sy - a[i].y;
		ret += (sqrt(dx * dx + dy * dy)) * a[i].w;
	}
	return ret;
}
void Simulate_Anneal() {
	double sx = ansx, sy = ansy, t = 1926;
	while(t > 1e-14) {
		double tx = ansx + (rand() * 2 - RAND_MAX) * t, ty = ansy + (rand() * 2 - RAND_MAX) * t;
		double now = get_energy(tx, ty), del = now - ans;
		if(del < 0) sx = tx, sy = ty, ansx = sx, ansy = sy, ans = now;
		else if(exp(-del / t) * RAND_MAX > rand()) sx = tx, sy = ty;
		t *= 0.993;
	}
}
// 莫队
/* 小z的袜子
小Z把这N只袜子从1到N编号，然后从编号L到R
(L 尽管小Z并不在意两只袜子是不是完整的一双，
甚至不在意两只袜子是否一左一右，他却很在意袜子的颜色，
毕竟穿两只不同色的袜子会很尴尬。)
你的任务便是告诉小Z，他有多大的概率抽到两只颜色相同的袜子。
当然，小Z希望这个概率尽量高，所以他可能会询问多个(L,R)以方便自己选择。
然而数据中有L=R的情况，请特判这种情况，输出0/1。
*/
struct md { int l, r, id; long long a, b; }ha[M];
long long gcd(long long a, long long b) {
	while(b ^= a ^= b ^= a %= b); return a;
}
int n, m, col[M], base, dat[M];
long long s[M], ans;
long long p2(long long x) { return x * x; }
int cmp(md x, md y) { return dat[x.l] == dat[y.l] ? x.r < y.r : x.l < y.l; }
int mmp(md x, md y) { return x.id < y.id; }
void update(int x, int del) {
	ans -= p2(s[col[x]]);
	s[col[x]] += del;
	ans += p2(s[col[x]]);
}
void work() {
	scanf("%d%d", &n, &m);
	base = sqrt(n);
	for(int i = 1; i <= n; ++i) {
		scanf("%d", col + i);
		dat[i] = i / base + 1;
	}
	for(int i = 1; i <= m; ++i) {
		int l, r;
		scanf("%d%d", &l, &r);
		ha[i] = (md) {l, r, i};
	}
	int l = 1, r = 0;
	std::sort(ha + 1, ha + m + 1, cmp);
	for(int i = 1; i <= m; ++i) {
		for(; l < ha[i].l; ++l) update(l, -1);
		for(; l > ha[i].l; --l) update(l - 1, 1);
		for(; r < ha[i].r; ++r) update(r + 1, 1);
		for(; r > ha[i].r; --r) update(r, -1);
		if(ha[i].l == ha[i].r) {
			ha[i].a = 0ll, ha[i].b = 1ll;
			continue;
		}
		ha[i].a = ans - (ha[i].r - ha[i].l + 1);
		ha[i].b = 1ll * (ha[i].r - ha[i].l + 1) * (ha[i].r - ha[i].l);
		long long g = gcd(ha[i].a, ha[i].b);
		ha[i].a /= g, ha[i].b /= g;
	}
	std::sort(ha + 1, ha + m + 1, mmp);
	for(int i = 1; i <= m; ++i)
		printf("%lld/%lld\n", ha[i].a, ha[i].b);
}
// 二维凸包周长
namespace sunrui {
	const double eps = 1e-9;
	struct vec { double x, y; };
	vec operator +(const vec a, const vec b) { return (vec) {a.x + b.x, a.y + b.y}; }
	vec operator -(const vec a, const vec b) { return (vec) {a.x - b.x, a.y - b.y}; }
	double dot(const vec a, const vec b) { return a.x * b.x + a.y * b.y; }
	double cross(const vec a, const vec b) { return a.x * b.y - a.y * b.x; }
	bool cmp(vec a, vec b) { return fabs(a.x - b.x) > eps ? a.x < b.x : a.y < b.y; }
	int dcmp(double a) { return fabs(a) < eps ? 0 : (a < 0 ? -1 : 1); }
	double dis(const vec a, const vec b) { return sqrt(dot(a - b, a - b)); }
	vec p[10001], s[10001];
	int n, t = 1;
	double ans = 0;
	void work() {
		scanf("%d", &n);
		for(int i = 1; i <= n; i++)
			scanf("%lf%lf", &p[i].x, &p[i].y);
		std::sort(p + 1, p + 1 + n, cmp);
		s[1] = p[1];
		for(int i = 2; i <= n; i++) {
			for(; t > 1 && dcmp(cross(s[t] - s[t - 1], p[i] - s[t -1 ])) <= 0; t--);
			s[++t] = p[i];
		}
		for(int i = 1; i < t; i++) ans += dis(s[i], s[i + 1]);
		int k = t;
		for(int i = n - 1; i >= 1; i--) {
			for(; t > k && dcmp(cross(s[t] - s[t - 1], p[i] - s[t - 1])) <= 0; t--);
			s[++t] = p[i];
		}
		for(int i = k; i < t; i++) ans += dis(s[i], s[i + 1]);
		ans += dis(s[1], s[t]);
		printf("%.2lf\n", ans);
	}
}
// fread快读
#define gr() (S == T && (T = (S = BB) + fread(BB, 1, 1 << 15, stdin), S == T) ? EOF : *S++)
char BB[(1 << 20) + 1], *S = BB, *T = BB;
inline int read() {
	char c; int f = 1;
	while(!isdigit(c = gr())) if(c == '-') f = -f;
	int x = c ^ 48;
	while(isdigit(c = gr())) x = (x << 3) + (x << 1) + (c ^ 48);
	return x * f;
}
// 子集枚举
	// 枚举S的子集
for(int i = s; i; i = (i - 1) & s) {
	// operations
}
	//枚举大小为r的子集
for(int s = (1 << r) - 1; s < (1 << n);) {
	// operations
	int x = s & -s, y = s + x;
	s = ((s & ~y) / x >> 1) | y;
}
// 排列生成
next_permutation(a + 1, a + n + 1);
// 高精度
struct bigInt {
	int len, a[10005];
	void clear() { memset(a, 0, sizeof(a)); }
	void set1() { len = a[1] = 1; }
}zpl;
bigInt operator *(bigInt a, int k) {
	bigInt h;
	h.clear();
	for(int i = 1; i <= a.len; ++i)
		h.a[i] = a.a[i] * k;
	for(int i = 2; i <= a.len; ++i) {
		h.a[i] += h.a[i - 1] / 10;
		h.a[i - 1] %= 10;
	}
	h.len = a.len;
	while(h.a[h.len] > 10) {
		h.a[h.len + 1] = h.a[h.len] / 10;
		h.a[h.len++] %= 10;
	}
	return h;
}
bigInt operator /(bigInt a, int k) {
	bigInt h;
	h.clear();
	int j = 0;
	for(int i = a.len; i > 0; --i) {
		j = j * 10 + a.a[i];
		h.a[i] = j / k;
		j %= k;
	}
	h.len = a.len;
	while(h.a[h.len] == 0) --h.len;
	return h;
}
void print(bigInt a) {
	for(int i = a.len; i > 0; --i)
		printf("%d", a.a[i]);
}
// Big Prime
1e9 + 7
19260817
19260823
998244353
// 归并排序
void merge_sort(int l, int r) {
	if(l >= r) return;
	int mid = l + r >> 1;
	merge_sort(l, mid), merge_sort(mid + 1, r);
	int i = l, j = mid + 1, k = l;
	while(i <= mid && j <= r) {
		if(a[i] > a[j]) b[k++] = a[j++], rev += (mid - i + 1); // 逆序对数
		else b[k++] = a[i++];
	}
	while(i <= mid) b[k++] = a[i++];
	while(j <= r) b[k++] = a[j++];
	for(i = l; i <= r; ++i)
		a[i] = b[i];
}
// 基数排序
int a[10000], cnt[65537], b[10000];
int f(int x, int p) {
	return (x >> (p * 16)) & 65536;
}
void radix_sort() {
	int *x = a, *y = b;
	for(int z = 0; z < 2; ++z) {
		for(int i = 0; i < 65536; ++i)
			cnt[i] = 0;
		for(int i = 0; i < n; ++i)
			++cnt[f(x[i], z)];
		for(int i = 1; i < 65536; ++i)
			cnt[i] += cnt[i - 1];
		for(int i = n - 1; i >= 0; --i)
			y[--cnt[f(x[i], z)]] = x[i];
		std::swap(x, y);
	}
}
// 模拟
	// BZOJ 1972 [SDOI2010] 猪国杀
#include <bits/stdc++.h>
int n, m, fnum, ed, now;
char hpai[2018];
struct pig {
	bool f, jmp, like, zhuge, usd[2018];
	int hp, num;
	char hd[2018];
}a[11];
bool MP(int k) { return k == 0 ? 1 : 0; }
bool ZP(int k) { return (a[k].f == 0 && k) ? 1 : 0; }
bool FP(int k) { return (a[k].f == 1) ? 1 : 0; }
void disca(int k) { a[k].num = a[k].zhuge = 0; }
void readpai() {
	scanf("%d%d", &n, &m);
	for(int i = 0; i < n; ++i) {
		char s[5];
		scanf("%s", s);
		if(s[0] == 'F') a[i].f = 1, ++fnum;
		for(int j = 1; j < 5; ++j)
			scanf(" %c", &a[i].hd[j]);
		a[i].num = a[i].hp = 4;
	}
	for(int i = 0; i < m; ++i)
		scanf(" %c", &hpai[i]);
	if(!fnum) ed = 1;
}
int dis(int x, int y) {
	int d = 1;
	for(int i = (x + 1) % n; i ^ y; i = (i + 1) % n)
		if(a[i].hp) ++d;
	return d;
}
void Reset(int k) {
	int tot = 0;
	for(int i = 1; i <= a[k].num; ++i) {
		if(a[k].usd[i]) continue;
		++tot;
		a[k].hd[tot] = a[k].hd[i];
		a[k].usd[tot] = 0;
	}
	a[k].num = tot;
}
void Showhand(int k) {
	Reset(k);
	for(int i = 1; i <= a[k].num; ++i) {
		printf("%c", a[k].hd[i]);
		if(i != a[k].num) putchar(' ');
	}
	putchar('\n');
}
void Gethand(int k, int num) {
	if(ed) return;
	for(int i = 1; i <= num; ++i) {
		++a[k].num;
		a[k].hd[a[k].num] = hpai[now];
		a[k].usd[a[k].num] = 0;
		if(m - 1 > now) ++now;
	}
}
void Kill(int k, int frm) {
	if(MP(k)) { ed = 2; return; }
	if(FP(k)) {
		--fnum;
		if(fnum == 0) ed = 1;
		Gethand(frm, 3);
	}
	if(ZP(k) && MP(frm)) disca(frm);
}
void Jump(int k) {
	a[k].jmp = 1;
}
int Useca(int k, char c) {
	for(int i = 1; i <= a[k].num; ++i)
		if(a[k].hd[i] == c && a[k].usd[i] == 0) {
			a[k].usd[i] = 1;
			return 1;
		}
	return 0;
}
int sha(int k) {
	return Useca(k, 'K');
}
int shan(int k) {
	return Useca(k, 'D');
}
int tao(int k) {
	return Useca(k, 'P');
}
int wuxie(int k) {
	if(Useca(k, 'J')) {
		Jump(k);
		return 1;
	}
	return 0;
}
void like(int k) {
	if(a[k].jmp == 0)
		a[k].like = 1;
}
void Wound(int k, int frm) {
	--a[k].hp;
	if(a[k].hp == 0) {
		if(tao(k)) ++a[k].hp;
		else Kill(k, frm);
	}
}
int Askwuxie(int k, int f) {
	int i = k;
	for(;;) {
		if(a[i].hp) if(a[i].f == f)
			if(wuxie(i)) {
				if(Askwuxie(i, f ^ 1) == 0) return 1;
				return 0;
			}
		i = (i + 1) % n;
		if(i == k) return 0;
	}
}
int Needwuxie(int k, int frm) {
	if(a[k].jmp == 0 && MP(k) == 0) return 0;
	if(Askwuxie(frm, a[k].f)) return 1;
	return 0;
}
void AOE(int k, int f) {
	for(int i = (k + 1) % n; i ^ k; i = (i + 1) % n)
		if(a[i].hp) {
			if(Needwuxie(i, k)) continue;
			if(f == 1) {
				if(sha(i) == 0) {
					if(MP(i)) like(k);
					Wound(i, k);
				}
			}
			if(f == 2) {
				if(shan(i) == 0) {
					if(MP(i)) like(k);
					Wound(i, k);
				}
			}
			if(ed) return;
		}
}
int Duel(int k, int frm) {
	if(ZP(k) && MP(frm)) return 1;
	if(Needwuxie(k, frm)) return 2; // (pre)return 0
	for(;;) {
		if(sha(k) == 0) return 1;
		if(sha(frm) == 0) return 0;
	}
}
int Attack(int k, int f) {
	if(FP(k)) {
		if(dis(k, 0) == 1 && f == 1) {
			Jump(k);
			if(shan(0) == 0) Wound(0, k);
			return 1;
		}
		if(f == 2) {
			Jump(k);
			int res = Duel(0, k);
			if(res == 1) Wound(0, k);
			else if(res == 0) Wound(k, 0);
			return 1;
		}
	}
	if(MP(k)) {
		for(int i = (k + 1) % n; i ^ k; i = (i + 1) % n) {
			if(a[i].hp) {
				if((FP(i) && a[i].jmp) || (a[i].like && !a[i].jmp)) {
					if(dis(k, i) == 1 && f == 1) {
						if(shan(i) == 0) Wound(i, k);
						return 1;
					}
					if(f == 2) {
						int res = Duel(i, k);
						if(res == 1) Wound(i, k);
						else if(res == 0) Wound(k, i);
						return 1;
					}
				}
			}
		}
	} else {
		for(int i = (k + 1) % n; i ^ k; i = (i + 1) % n) {
			if(a[i].hp) {
				if((a[k].f ^ a[i].f) && (a[i].jmp)) {
					if(dis(k, i) == 1 && f == 1) {
						Jump(k);
						if(shan(i) == 0) Wound(i, k);
						return 1;
					}
					if(f == 2) {
						Jump(k);
						int res = Duel(i, k);
						if(res == 1) Wound(i, k);
						else if(res == 0) Wound(k, i); //(pre) if(res =- 0)
						return 1;
					}
				}
			}
		}
	}
	return 0;
}
void Move(int k) {
	Reset(k);
	Gethand(k, 2);
	int flag = 0;
	for(int i = 1; i <= a[k].num; ++i) {
		if(ed || !a[k].hp) return;
		if(a[k].usd[i] == 1) continue;
		if(a[k].hd[i] == 'P' && a[k].hp < 4)
			a[k].usd[i] = 1, ++a[k].hp;
		else if(a[k].hd[i] == 'N')
			a[k].usd[i] = 1, AOE(k, 1), i = 0;
		else if(a[k].hd[i] == 'W')
			a[k].usd[i] = 1, AOE(k, 2), i = 0;
		else if(a[k].hd[i] == 'K' && (!flag || a[k].zhuge)) {
			if(Attack(k, 1)) {
				a[k].usd[i] = 1;
				flag = 1;
				i = 0;
			}
		} else if(a[k].hd[i] == 'F') {
			if(Attack(k, 2))
				a[k].usd[i] = 1, i = 0;
		} else if(a[k].hd[i] == 'Z') {
			a[k].usd[i] = 1;
			if(!a[k].zhuge) a[k].zhuge = 1, i = 0;
		}
	}
}
void solve() {
	for(int i = 0; i < n; ++i)
		if(a[i].hp) Move(i);
}
void print() {
	if(ed == 1) puts("MP");
	else if(ed == 2) puts("FP");
	for(int i = 0; i < n; ++i) {
		if(a[i].hp == 0) puts("DEAD");
		else Showhand(i);
	}
}
int main() {
	readpai();
	while(!ed) solve();
	print();
	return 0;
}