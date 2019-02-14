// Luogu Online Judge P3690 [template] Link-Cut Tree
// status: Accepted
#pragma GCC optimize(2)
#include <bits/stdc++.h>
// =====FAST IO START=====
#define gr() (S==T && (T=(S=BB)+fread(BB,1,1<<15,stdin),S==T) ? EOF : *S++)
char BB[1<<20],*S=BB,*T=BB;
inline int read() {
	char c;
	while(!isdigit(c=gr()));
	int x=c^48;
	while(isdigit(c=gr())) x=(x<<3)+(x<<1)+(c^48);
	return x;
}
#undef gr
char pbuf[10000000],*pp=pbuf;
inline void write(int x) {
	static int st[35];
	register int top=0;
	if(!x) st[++top]=0;
	while(x) st[++top]=x%10,x/=10;
	while(top) *pp++=st[top--]^48;
}
// =====FAST IO READ=====
const int N=3e5+10;
#define swap(x,y) x^=y,y^=x,x^=y
#define fa(x) tr[x].f
#define ls(x) tr[x].c[0]
#define rs(x) tr[x].c[1]
int v[N],st[N];
struct link_cut_tree {
	int f,c[2],s,r;									// f->father c->children s->sum r->reverse tag
}tr[N];
// =====SPALY START=====
inline int idt(int x) {								// get is lson or rson
	return tr[fa(x)].c[0]==x?0:1;
}
inline void connect(int x,int fa,int to) {			// connect function of spaly
	tr[x].f=fa;
	tr[fa].c[to]=x;
}
inline int isrt(int x) {							// check if it is a root of a spaly
	return ls(fa(x))!=x && rs(fa(x))!=x;			// for a root, it has a edge to its father, but its father has no edge to it
}
inline void update(int x) {
	tr[x].s=tr[ls(x)].s^tr[rs(x)].s^v[x];			// update xor sum
}
inline void push_down(int x) {						// reverse tag push down
	if(tr[x].r) {
		swap(ls(x),rs(x));
		tr[ls(x)].r^=1;
		tr[rs(x)].r^=1;
		tr[x].r=0;
	}
}
inline void rotate(int x) {							// rotate function of spaly
	int yy=tr[x].f,rr=tr[yy].f,yys=idt(x),rrs=idt(yy);
	int bb=tr[x].c[yys^1];
	tr[x].f=rr;
	if(!isrt(yy)) connect(x,rr,rrs);
	connect(bb,yy,yys);
	connect(yy,x,yys^1);
	update(yy),update(x);
}
inline void spaly(int x) {							// spaly function of spaly
	int y=x,tp=0;
	st[++tp]=y;
	while(!isrt(y)) st[++tp]=y=fa(y);
	while(tp) push_down(st[tp--]);
	for(int y=fa(x);!isrt(x);rotate(x),y=fa(x))
		if(!isrt(y))
			rotate(idt(x)==idt(y) ? x :y);
}
// =====SPALY END=====
// =====LCT START=====
inline void access(int x) {							// access function of Link-Cut Tree, make root -> x real path, make x -> it's sons vitrual path
	for(int y=0;x;x=fa(y=x))						// make x root, y is x's son, make y x's rson(make x -> y real path), update information
		spaly(x),rs(x)=y,update(x);
}
inline void make_root(int x) {						// make x the root of the original tree
	access(x);
	spaly(x);
	tr[x].r^=1;										// update reverse tag
	push_down(x);
}
inline int get_root(int x) {						// find x's root in the original tree
	access(x),spaly(x);
	push_down(x);
	while(ls(x)) push_down(x=ls(x));				// find the node with the smallest depth
	return x;
}
inline void split(int x,int y) {					// get a path x -> y
	make_root(x);									// make x the root of the original tree
	access(y);										// query path x -> y
	spaly(y);										// make y the root, then the chain is x -> y
}
inline void link(int x,int y) {						// link function
	make_root(x);									// make x the root of the original tree
	if(get_root(y)!=x) fa(x)=y;						// if x and y in different spalys, make y x's father (Link)
}
inline void cut(int x,int y) {						// cut function
	make_root(x);									// make x the root of the original tree
	if(get_root(y)==x && fa(x)==y && ls(y)==x && !rs(x))
		fa(x)=tr[y].c[0]=0,update(y);				// if x and y in the same spaly, and x is y's lson, and x doesn't have rson, cut the path x -> y
}
// =====LCT END=====
int main() {
	int n=read(),m=read();
	for(register int i=1;i<=n;i++)
		v[i]=read();
	for(register int i=1,opt,x,y;i<=m;i++) {
		opt=read();
		x=read();
		y=read();									// opt = 0, get the path x -> y, print the xor sum
		if(!opt) split(x,y),write(tr[y].s),*pp++='\n';
		if(opt==1) link(x,y);						// opt = 1, link (x,y)
		if(opt==2) cut(x,y);						// opt = 2, cut (x,y)
		if(opt==3) spaly(x),v[x]=y;					// opt = 3, make x the root, update the v[] of x
	}
	fwrite(pbuf,1,pp-pbuf,stdout);
	return 0;
}