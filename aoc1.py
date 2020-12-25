_='''1'''
l=map(int,_.split());a=[(0,)]+[()]*9999
for i in l:
    for j in range(2020,-1,-1):
        if a[j]:
            a[i+j]=a[j]+(i,);b=a[i+j]
            if i+j==2020 and len(b)==4:print(b[1]*b[2]*b[3])
_='''2'''
a=list(i.split()for i in _.split('\n'));t=0
for i in a:
    l,h=map(int,i[0].split('-'));c=i[1][0];s=i[2]
    if (s[l-1]==c)+(s[h-1]==c)==1:t+=1
print(t)
_='''3'''
a=_.split();p=1;s=[(1,1),(3,1),(5,1),(7,1),(1,2)]
for j in s:
    t=0
    for i in range(len(a)//j[1]):
        if a[i*j[1]][(j[0]*i)%len(a[0])]=='#':t+=1
    p*=t
print(p)
_='''4'''
a=[i.split()for i in _.split('\n\n')];z='0123456789abcdef'
t=['byr','iyr','eyr','hgt','hcl','ecl','pid','cid'];r=0
for i in a:
    u=[0]*8;s=1
    for j in i:
        v=t.index(j[:3]);u[v]+=1
        if v==0 and not 1920<=int(j[4:])<=2002:s=0;break
        elif v==1 and not 2010<=int(j[4:])<=2020:s=0;break
        elif v==2 and not 2020<=int(j[4:])<=2030:s=0;break
        elif v==3:
            if j[-2:]=='cm' and not 150<=int(j[4:-2])<=193:s=0;break
            elif j[-2:]=='in' and not 59<=int(j[4:-2])<=76:s=0;break
            else:s=0;break
        elif v==4:
            if j[4]=='#'and all(j[i]in z for i in range(5,len(j))):pass
            else:s=0;break
        elif v==5:
            if j[4:]in['amb','blu','brn','gry','grn','hzl','oth']:pass
            else:s=0;break
        elif v==6:
            if len(j)==13 and all(j[i]in z[:10]for i in range(5,13)):pass
            else:s=0;break
    if all(u[:7])and s:r+=1
print(r)
_='''5'''
a=_.split();r=set()
for i in a:
    n=eval('0b'+i[:-3].replace('F','0').replace('B','1'))
    m=eval('0b'+i[-3:].replace('R','1').replace('L','0'))
    r.add(8*n+m)
print(sorted(set(range(1024))-r))
_='''6'''
a=_.split('\n\n');t=0
for i in a:
    s=[0]*26;b=i.split()
    for j in range(len(b)):
        for c in b[j]:s[ord(c)-97]+=1
    for j in s:
        if j==len(b):t+=1
print(t)
_='''7'''
x=_.split('\n');e={}
for i in x:
    a,b=i[:-1].split(' contain ');l=b.split(' bag')[:-1]
    for j in range(len(l)):
        if l[j][:3]=='s, ':l[j]=l[j][3:]
        elif l[j][:2]==', ':l[j]=l[j][2:]
    a=a[:-5]
    if a not in e:e[a]=[]
    for i in l:
        if i[2]!=' ':e[a].append((int(i[0]),i[2:]))
q=[(1,'shiny gold')];nq=[];t=0
while q:
    m,v=q.pop()
    for o,i in e.get(v,[]):nq.append((o*m,i));t+=o*m
    if not q:q=nq;nq=[];print(q)
print(t)
_='''8'''
a=_.split('\n')
for j in range(len(a)):
    b=a[:];d=set();v=i=0
    if a[j][0]=='n':b[j]='j'+b[j]
    elif a[j][0]=='j':b[j]='n'+b[j]
    else:continue
    while i not in d and i<len(a):
        d.add(i);x,y=b[i].split();y=int(y)
        if x[0]=='a':v+=y;i+=1
        elif x[0]=='j':i+=y
        else:i+=1
    if i==len(a):print(v)
_='''9'''
a=list(map(int,_.split()));n=104054607;s=[0]*len(a)
for i in range(len(a)):
    for j in range(i+1):s[j]+=a[i]
    if n in s[:i]:l=a[s.index(n):i+1];print(min(l)+max(l))
_='''10'''
a=sorted(map(int,_.split()));a+=[max(a)+3];d=[1]+[0]*max(a)
for i in a:d[i]=d[i-1]+d[i-2]+d[i-3]
print(d[max(a)])
_='''11'''
import copy,itertools
l=[*itertools.product((0,1,-1),(0,1,-1))][1:]
a=[*map(list,_.split())];n=len(a);m=len(a[0]);b=[]
while b!=a:
    b=copy.deepcopy(a)
    for x,y in itertools.product(range(n),range(m)):
        c=0
        for i,j in l:
            v=1
            while 0<=x+v*i<n and 0<=y+v*j<m and b[x+v*i][y+v*j]=='.':v+=1
            if 0<=x+v*i<n and 0<=y+v*j<m and b[x+v*i][y+v*j]=='#':c+=1
        if b[x][y]=='L'and c<1:a[x][y]='#'
        if b[x][y]=='#'and c>4:a[x][y]='L'
print(sum(i.count('#')for i in a))
_='''12'''
a=_.split();x,y=0,0;dx,dy=10,1
for i in a:
    if i[0]=='N':dy+=int(i[1:])
    if i[0]=='S':dy-=int(i[1:])
    if i[0]=='E':dx+=int(i[1:])
    if i[0]=='W':dx-=int(i[1:])
    if i[0]=='L':
        for j in range(int(i[1:])//90):dx,dy=-dy,dx
    if i[0]=='R':
        for j in range(int(i[1:])//90):dx,dy=dy,-dx
    if i[0]=='F':
        for j in range(int(i[1:])):x,y=x+dx,y+dy
print(abs(x)+abs(y))
_='''13'''
def gcd(x, y):
    while y:x, y = y, x % y
    return x
def extended_gcd(a, b):
    s, old_s, r, old_r = 0, 1, b, a
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return old_r, old_s, (old_r - old_s * a) // b if b else 0
def composite_crt(b, m):
    x, m_prod = 0, 1
    for bi, mi in zip(b, m):
        g, s, _ = extended_gcd(m_prod, mi)
        if ((bi - x) % mi) % g:return None
        x += m_prod * (s * ((bi - x) % mi) // g)
        m_prod = (m_prod * mi) // gcd(m_prod, mi)
    return x % m_prod
n,a=_.split();n=int(n);a=a.split(',');b=[];m=[]
for i in range(len(a)):
    if a[i]!='x':m.append(int(a[i]));b.append(-i)
print(composite_crt(b,m))
_='''14'''
import copy
a=_.split('\n');t=0;s={}
for i in a:
    if i[1]=='a':m=i.split()[2]
    else:
        b,_,e=i.split();c=bin(int(b[4:-1]))[2:];c=[[*c.zfill(36)]]
        for j in range(36):
            if m[j]=='1':
                for k in c:k[j]='1'
            elif m[j]=='X':
                c=copy.deepcopy(c)+copy.deepcopy(c)
                for k in range(len(c)//2):c[k][j]='0'
                for k in range(len(c)//2,len(c)):c[k][j]='1'
        l=[int(''.join(k),2)for k in c]
        for v in l:s[v]=int(e)
print(sum(s[i]for i in s))

_='''15'''
l=[0,14,1,3,7,9];u={0:0,14:1,1:2,3:3,7:4,9:5}
while len(l)<3*10**7:
    if l[-1] in u and u[l[-1]]!=len(l)-1:
        l.append(len(l)-1-u[l[-1]]);u[l[-2]]=len(l)-2
    else:l.append(0)
    u[l[-2]]=len(l)-2
print(l[-1])
_='''16'''
a,b,c=_.split('\n\n');d=[];t=0
for i in a.split('\n'):
    v1,v2,v4=i.split('-');v1=v1.split(' ')[-1];v2,v3=v2.split(' or ')
    d.append(list(map(int,(v1,v2,v3,v4))))
p=[list(range(len(d)))for i in range(len(d))]
for i in c.split('\n')[1:]:
    v=list(map(int,i.split(',')));z=1;bad=[]
    for j in range(len(v)):
        bad.append([]);t=0
        for x in p[j]:
            if d[x][0]<=v[j]<=d[x][1] or d[x][2]<=v[j]<=d[x][3]:t+=1
            else:bad[-1].append(x)
        if not t:z=0
    if z:
        for j in range(len(bad)):
            for x in bad[j]:p[j].remove(x)
p.append([]);g=sorted((len(p[i]),set(p[i]),i)for i in range(21))
for i in range(1,21):print(g[i][1]-g[i-1][1],g[i][2])
_='''17'''
import copy;from itertools import product as P;Z=range
B=7;a=list(map(list,_.split('\n')));L=len(a)
for i in Z(L):a[i]=list('.'*B)+a[i]+list('.'*B)
a=[list('.'*(B+L+B)) for i in Z(B)]+a+[list('.'*(B+L+B))for i in Z(B)]
a=([[list('.')*(B+L+B)for i in Z(B+L+B)]for j in Z(B)]+[a]+
   [[list('.')*(B+L+B)for i in Z(B+L+B)]for j in Z(B)])
a=([[[list('.')*(B+L+B)for i in Z(B+L+B)]for j in Z(B+B+1)]for k in Z(B)]+[a]+
   [[[list('.')*(B+L+B)for i in Z(B+L+B)]for j in Z(B+B+1)]for k in Z(B)])
b=copy.deepcopy(a);l=list(itertools.product((0,-1,1),repeat=4))[1:];r=0
for i in Z(B-1):
    for w,x,y,z in P(Z(1,2*B),Z(1,2*B),Z(1,B+L+B-1),Z(1,B+L+B-1)):
        t=0
        for dw,dx,dy,dz in l:
            if a[w+dw][x+dx][y+dy][z+dz]=='#':t+=1
        if a[w][x][y][z]=='#' and (t<2 or t>3):b[w][x][y][z]='.'
        elif a[w][x][y][z]=='.' and t==3:b[w][x][y][z]='#'
    a=copy.deepcopy(b)
for w,x,y,z in P(Z(1,2*B),Z(1,2*B),Z(1,B+L+B-1),Z(1,B+L+B-1)):
    if a[w][x][y][z]=='#':r+=1
print(r)
_='''18'''
a=_.split('\n');t=0
for i in a:
    s=[''];i='('+i+')';i=i.replace('*',',');i=i.replace(')',',)')
    for j in i:
        if j=='(':s.append('(')
        elif j==')':
            v=eval(s[-1]+')');p=1
            for k in v:p*=k
            s[-2]+=str(p);s.pop()
        else:s[-1]+=j
    t+=int(s[0])
print(t)
_='''19'''
a,b=_.split('\n\n');a=a.split('\n');b=b.split('\n');g={};t=0
Y=lambda x:(*map(int,x.split()),)
for i in a:
    n,j=i.split(': ');n=int(n)
    if '|' in j:k,l=j.split(' | ');k=Y(k);l=Y(l);g[n]=(k,l)
    elif '"' in j:k=j[1];g[n]=(k,)
    else:k=Y(j);g[n]=(k,)
g[8]=((42,),(42,8));g[11]=((42,31),(42,11,31))
def sol(msg,rule,end=0):
    pos=set()
    for i in g[rule]:
        if type(i)==str:
            if msg and msg[0]==i:pos={msg[1:]}
            break
        copy=msg;v=sol(copy,i[0])
        for z in range(1,len(i)):
            w=set()
            for j in v:w=w.union(sol(j,i[z]))
            v=w
        pos=pos.union(v)
    if end:return""in pos
    return pos
for i in b:t+=sol(i,0,end=1)
print(t)
_='''20'''
s=['                  # ','#    ##    ##    ###',' #  #  #  #  #  #   '];S=set()
from itertools import product as P;from random import choice as C;Z=range
for x,y in P(Z(3),Z(20)):
    if s[x][y]=='#':S.add((x,y))
a=_.split('\n\n');s={};c={};T={};r=1
X=lambda s:tuple(sorted((s,s[::-1])))
V=lambda t,s:''.join(t[i][s]for i in Z(10))
H=lambda t,s:''.join(t[s])
L=lambda i:[X(v)for v in[H(i,0),H(i,-1),V(i,0),V(i,-1)]]
R=lambda M:[''.join(M[-1-x][y]for x in Z(len(M)))for y in Z(len(M))]
F=lambda M:[s[::-1]for s in M]
G=lambda M:sum(i.count('#')for i in M)
for i in a:
    i=i.split('\n');t,i=i[0],i[1:];t=int(t.split(' ')[1][:-1]);T[t]=i
    for n in L(i):s[n]=s.get(n,set()).union({t});c[n]=c.get(n,0)+1
for t in T:
    if sum(c[n]for n in L(T[t]))<7:x=T[t];y=t;break
b=[[0]*12 for i in Z(12)];d=[[0]*12 for i in Z(12)]
while c[X(H(x,0))]+c[X(V(x,0))]>2:x=R(x)
b[0][0]=x;d[0][0]=y;B=[[0]*96 for i in Z(96)];E=[[0]*96 for i in Z(96)]
for x,y in P(Z(12),Z(12)):
    if y:
        d[x][y]=(s[X(V(b[x][y-1],-1))]-{d[x][y-1]}).pop();b[x][y]=T[d[x][y]]
        while V(b[x][y],0)!=V(b[x][y-1],-1):b[x][y]=C((R,F))(b[x][y])
    elif x:
        d[x][0]=(s[X(H(b[x-1][0],-1))]-{d[x-1][0]}).pop();b[x][0]=T[d[x][0]]
        while H(b[x][0],0)!=H(b[x-1][0],-1):b[x][0]=C((R,F))(b[x][0])
for w,x,y,z in P(Z(12),Z(12),Z(8),Z(8)):B[w*8+y][x*8+z]=b[w][x][y+1][z+1]
while 1:
    B=C((R,F))(B)
    for x,y in P(Z(93),Z(76)):
        if all(B[x+a][y+b]=='#'for a,b in S):
            for a,b in S:E[x+a][y+b]='#'
    if G(E):print(G(B)-G(E));break
_='''21'''
a=_.split('\n');n=len(a);b=[];c=[];d=set();f={};k=set()
for i in a:
    l,p=i.split(' (contains ');l=set(l.split(' '));p=set(p[:-1].split(', '))
    b.append(l);c.append(p);d=d.union(p);k=k.union(l)
d=list(d);k=list(k)
for i in d:
    e=set(k)
    for j in range(n):
        if i in c[j]:e=e.intersection(b[j])
    f[i]=e
R={'shellfish':'xcfpc','nuts':'spbxz','peanuts':'pfdkkzp','dairy':'gpgrb',
   'fish':'gtjmd','eggs':'tjlz','soy':'txzv','wheat':'znqbr'}
print(','.join(i[1] for i in sorted((i,R[i])for i in R)))
_='''22'''
import copy
a,b=_.split('\n\n');a=a.split('\n')[1:];b=b.split('\n')[1:]
a=tuple(map(int,a));b=tuple(map(int,b));l=0
def r(a,b):
    d=set();global l
    while a and b:
        l+=1
        if l%10000<1:print(l)
        if(a,b)in d:return(1,a)
        d.add((a,b))
        if a[0]<len(a) and b[0]<len(b):w=r(a[1:a[0]+1],b[1:b[0]+1])[0]
        else:w=1+(a[0]<b[0])
        if w==1:a=a[1:]+(a[0],b[0]);b=b[1:]
        else:b=b[1:]+(b[0],a[0]);a=a[1:]
    if a:return(1,a)
    else:return(2,b)
w=r(a,b)[1];t=0
for i in range(len(w)):t+=(len(w)-i)*w[i]
print(t)
_='''23'''
class Node:
    def __init__(self, value):
        self.value=value;self.next=None;self.prev=None;self.pred=None
class LL:
    def __init__(self):
        self.z=Node(None);self.z.next=self.z;self.z.prev=self.z
    def append(self,value):
        return self.insert_between(Node(value),self.z.prev,self.z)
    def insert_between(self,n,l,r):
        n.prev=l;n.next=r;l.next=n;r.prev=n
        return n
s=[*map(int,_)];l=LL()
for i in s:l.append(i)
for i in range(2,10):
    w=l.z.next
    while w.value!=i:w=w.next
    x=l.z.next
    while x.value!=i-1:x=x.next
    w.pred=x
p=l.z.next
for i in range(10,10**6+1):v=l.append(i);v.pred=p;p=v
w=l.z.next
while w.value!=1:w=w.next
w.pred=p;w=l.z.next
for _ in range(10**7):
    if _ and _%(10**5)<1:print(_)
    skip=0;a=w.next
    if a==l.z:a=a.next;skip=1
    b=a.next
    if b==l.z:b=b.next
    c=b.next
    if c==l.z:c=c.next
    d=c.next
    if d==l.z:d=d.next;skip=1
    L=(a.value,b.value,c.value);f=w.pred
    while f.value in L:f=f.pred
    c.next=f.next;f.next.prev=c;f.next=a;a.prev=f
    if skip:
        w.next=l.z;l.z.prev=w;d.prev=l.z;l.z.next=d
    else:w.next=d;d.prev=w
    w=w.next
    if w==l.z:w=w.next
w=l.z.next
while w.value!=1:w=w.next
a=w.next
if a==l.z:a=a.next;skip=1
b=a.next
if b==l.z:b=b.next
print(a.value*b.value)
_='''24'''
import copy
a=_.split();c={};d=[(1,0),(0,-1),(-1,-1),(-1,0),(0,1),(1,1),(0,0)]
r=[('se','1'),('sw','2'),('nw','4'),('ne','5'),('e','0'),('w','3')]
for i in a:
    for j in r:i=i.replace(j[0],j[1])
    s=[0,0]
    for j in i:dx,dy=d[int(j)];s[0]+=dx;s[1]+=dy
    c[tuple(s)]=1-c.get(tuple(s),0)
for D in range(100):
    e={};s=set(sum(([(i[0]+j[0],i[1]+j[1])for j in d]for i in c),[]))
    for i in s:
        t=sum(c.get((i[0]+d[j][0],i[1]+d[j][1]),0)for j in range(6))
        if(c.get(i,0)and 0<t<3)or(c.get(i,0)<1 and t==2):e[i]=1
    c=copy.copy(e)
print(len(e))
_='''25'''
a=17773298;b=15530095;M=20201227
for i in range(M):
    if pow(7,i,M)==a:break
for j in range(M):
    if pow(7,j,M)==b:break
print(pow(7,i*j,M))
