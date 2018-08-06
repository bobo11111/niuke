package com.hust;

import java.util.*;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; next = null; }
}

public class Solution {
    private ArrayList<ArrayList<String>> ret = new ArrayList<>();
    private  ArrayList<String> cur  = new ArrayList<>();
    public ArrayList<ArrayList<String>> partition(String s) {
        if(s == null)
            return ret;
        if(s .equals("") ){
            ret.add(new ArrayList<>(cur));

            return ret;
        }
        for(int i = 1; i<s.length()+1 ; i++) {
            String m = s.substring(0,i);
            if(Ispalindrome(m)) {
                cur.add(m);
                partition(s.substring(i,s.length()));
                cur.remove(cur.size()-1);
            }
        }
        return ret;
    }

    private boolean Ispalindrome(String a) {
        if(a == null || a.length()<2)
            return true;
        char [] reverse = a.toCharArray();
        int len = reverse.length;
        for(int i = 0 ; i<len/2;i++) {
            if(reverse[i] != reverse[len-1-i])
                return false;
        }
        return true;
    }

    public void solve(char[][] board) {
        if( board == null || board.length==0 || board[0].length == 0 )
            return ;
        int rows = board.length;
        int cols = board[0].length;
        for(int i = 0 ; i<rows ; i++) {
            if(board[i][0] == 'O')
                dfs(board,i,0);
            if(board[i][cols-1] == 'O')
                dfs(board , i , cols-1);
        }
        for(int i = 1 ; i<=cols-2 ; i++) {
            if(board[0][i] == 'O')
                dfs(board , 0 , i);
            if(board[rows-1][i] == 'O')
                dfs(board , rows-1 , i);
        }
        for(int i = 0 ; i<rows ; i++)
            for(int j = 0 ; j<cols ;j++) {
                if(board[i][j] == '*')
                    board[i][j] = 'O';
                else if(board[i][j] == 'O')
                    board[i][j] = 'X';
            }

    }
    public void dfs(char [][] board , int rows , int cols) {
        board[rows][cols] = '*';
        if(rows-1>0 && board[rows-1][cols] == 'O')
            dfs(board , rows-1,cols);
        if(rows+1<board.length-1 && board[rows+1][cols] == 'O')
            dfs(board,rows+1,cols);
        if(cols-1>0 && board[rows][cols-1] == 'O')
            dfs(board , rows , cols-1);
        if(cols+1 <board[0].length-1 && board[rows][cols+1] == 'O')
            dfs(board , rows , cols+1);
    }

    public int ladderLength(String start, String end, HashSet<String> dict) {
        Queue<String> que = new LinkedList<>();
        int length = 1;
        que.offer(start);

        while(!que.isEmpty()) {
            int size = que.size();
            while(size-->0) {
                char[] cur = que.poll().toCharArray();
                for (int i = 0; i < cur.length; i++) {
                    char ch = cur[i];
                    for (char j = 'a'; j <= 'z'; j++) {
                        cur[i] = j;
                        String nString = String.valueOf(cur);
                        if (nString.equals(end))
                            return ++length;
                        if(dict.contains(nString)) {
                            que.offer(nString);
                            dict.remove(nString);
                        }
                    }
                    cur[i] = ch;
                }
            }
            length++;
        }
        return 0;

    }

    public boolean isPalindrome(String s) {
        if(s == null || s.equals(""))
            return true;
        s= s.toLowerCase();
        int i = 0 , j = s.length()-1;
        while(i<j) {
            char left = s.charAt(i);
            char right = s.charAt(j);
            boolean l = !(left>='a'&& left<='z' || left>='0' && left <='9');
            boolean r = !(right>='a'&& right<='z' || right>='0' && right <='9');
            if(l || r) {
                if(l)
                    i++;
                if(r)
                    j--;
                continue;
            }
            if(left != right)
                return false;
            i++;
            j--;

        }
        return true;
    }

    private int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if(root == null)
            return 0;
        maxSum(root);
        return max;
    }
    public int maxSum(TreeNode root) {
        if(root == null)
            return 0;
        int left = Math.max(0,maxSum(root.left));
        int right = Math.max(0 , maxSum(root.right));
        max = Math.max(max , left+right+root.val);
        return Math.max(left+root.val , right+root.val);
    }

    public int maxProfit(int[] prices) {
        if(prices == null || prices.length <2)
            return 0;
        int [] profit = new int[prices.length-1];
        int [] dp1 = new int[prices.length-1];
        int [] dp2 = new int[prices.length-1];
        for(int i = 1;i<prices.length;i++)
            profit[i-1] = prices[i] - prices[i-1];
        int max = 0;
        int leftmax = 0;
        for(int i = 0 ; i < profit.length ; i++){
            leftmax+=profit[i];
            if(leftmax>max)
                max = leftmax;
            if(leftmax < 0 )
                leftmax = 0;
            dp1[i] = max;
        }
        if(dp1[profit.length-1] == 0)
            return 0;
        int rightmax = 0;
        int rmax = 0;
        for(int i = profit.length-1;i>=0;i--) {
            rightmax += profit[i];
            if(rightmax>rmax)
                rmax = rightmax;
            if(rightmax<0)
                rightmax = 0;
            dp2[i] = rmax;
        }
        for(int i = 0;i<profit.length-1;i++)
            max = Math.max(max , dp1[i]+dp2[i+1]);
        return max;
    }

    public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
        if(triangle == null)
            return 0;
        int max = Integer.MAX_VALUE;
        int length = triangle.size()-1;
        for(int i = length ; i>=0 ;i--)
            max = Math.min(max,minTotal(triangle , length , i));
        return max;
    }
    public int minTotal(ArrayList<ArrayList<Integer>> triangle, int i , int j) {
        if(i == 0)
            return triangle.get(0).get(0);
        if(j == 0)
            return triangle.get(i).get(0)+minTotal(triangle , i-1 , 0);
        if(j == i)
            return triangle.get(i).get(j) +minTotal(triangle , i-1 , j-1);
        return triangle.get(i).get(j)+Math.min(minTotal(triangle , i-1,j-1),minTotal(triangle , i-1,j));
    }
    public ArrayList<Integer> getRow(int rowIndex) {
        ArrayList<Integer> list = new ArrayList<>(rowIndex+1);
        list.add(1);
        if(rowIndex == 0)
            return list;
        long tmp = 1;
        for(int i = 1; i<rowIndex ; i++) {
            tmp =tmp*(rowIndex+1-i)/i;
            list.add((int)tmp);
        }
        list.add(1);
        return list;
    }

    private int num;
    public int numDistinct(String S, String T) {
        if(S == null || T == null ||S.length()<T.length())
            return 0;
        numCount(S,T);
        return num;
    }
    public void numCount(String s , String t) {
        if(s.length()<t.length())
            return ;
        if(t.equals("")){
            num++;
            return;
        }
        for(int j = 0;j<s.length()&&s.length()-j>=t.length();j++) {
            if(s.charAt(j) == t.charAt(0))
                numCount(s.substring(j+1) , t.substring(1));
        }
    }


    public int minCut(String s) {
        if(s == null)
            return 0;
        int dp [] = new int[s.length()];
        boolean [][] p  = new boolean [s.length()][s.length()];
        Arrays.fill(dp,Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i = 0;i<s.length();i++)
            p[i][i] = true;
        for(int i = 0 ; i<s.length() ; i++)
            for(int j = 0 ;j<=i ; j++) {
                p[j][i] = s.charAt(i) == s.charAt(j) && (i-j<=2 || p[j+1][i-1]);
                if(p[j][i] == true)
                    dp[i] = Math.min(dp[i] , j == 0 ? 0 : dp[j-1]+1);
            }
        return dp[s.length()-1];
    }

    public TreeNode sortedListToBST(ListNode head) {
        if(head == null)
            return null;
        ListNode pre = findMid(head);
        ListNode mid = null;
        ListNode right = null;
        if(pre == null) {
            mid = head;
            right = mid.next;
            TreeNode cur = new TreeNode(mid.val);
            if(right!=null)
                cur.right = new TreeNode(right.val);
            return cur;

        }
        mid = pre.next;
        right = mid.next;
        mid.next = null;
        pre.next = null;
        TreeNode cur = new TreeNode(mid.val);
        cur.left = sortedListToBST(head);
        cur.right = sortedListToBST(right);
        return cur;

    }
    public ListNode findMid(ListNode head) {
        if(head == null)
            return null;
        ListNode low = head , high = head , pre = null;

        while(high.next!=null && high.next.next != null) {
            pre = low;
            low = low.next;
            high = high.next.next;
        }
        return pre;
    }

    public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null)
            return list;
        queue.offer(root);
        while(!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while(size--!=0) {
                TreeNode cur = queue.poll();
                tmp.add(cur.val);
                if(cur.left!=null)
                    queue.offer(cur.left);
                if(cur.right!=null)
                    queue.offer(cur.right);
            }
            list.add(0,new ArrayList<>(tmp));
        }
        return list;
    }

    //Given inorder and postorder traversal of a tree, construct the binary tree
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if(inorder == null || postorder == null)
            return null;
        return dTree(inorder,postorder,0,inorder.length-1,0,postorder.length-1);
    }
    public TreeNode dTree(int[] inorder , int[] postorder , int i,int j, int m,int n) {
        if(i==j || m == n)
            return new TreeNode(postorder[m]);
        if(j<i || n<m )
            return null;
        TreeNode node = new TreeNode(postorder[n]);
        int index = i;
        while(index <= j) {
            if(inorder[index] == postorder[n])
                break;
            index++;
        }
        node.left = dTree(inorder,postorder,i,index-1,m,m+index-i-1);
        node.right = dTree(inorder,postorder,index+1,j,m+index-i,n-1);
        return node;
    }

    //Given preorder and inorder traversal of a tree, construct the binary tree.
    public TreeNode buildTree2(int[] preorder, int[] inorder) {
        if(inorder == null || preorder == null)
            return null;
        return dTree2(inorder,preorder,0,inorder.length-1,0,preorder.length-1);
    }
    public TreeNode dTree2(int[] inorder , int[] preorder , int i,int j, int m,int n) {
        if(i==j || m == n)
            return new TreeNode(preorder[m]);
        if(j<i || n<m )
            return null;
        TreeNode node = new TreeNode(preorder[m]);
        int index = i;
        while(index <= j) {
            if(inorder[index] == preorder[m])
                break;
            index++;
        }
        node.left = dTree2(inorder,preorder,i,index-1,m+1,m+index-i);
        node.right = dTree2(inorder,preorder,index+1,j,m+index-i+1,n);
        return node;
    }

    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null)
            return list;
        queue.offer(root);
        int level = 1;
        while(!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while(size--!=0) {
                TreeNode cur = queue.poll();
                if(level%2 == 1)
                    tmp.add(cur.val);
                else
                    tmp.add(0,cur.val);
                if(cur.left!=null)
                    queue.offer(cur.left);
                if(cur.right!=null)
                    queue.offer(cur.right);
            }
            list.add(new ArrayList<>(tmp));
            level++;
        }
        return list;
    }

    public boolean isSymmetric(TreeNode root) {
        if(root == null)
            return true;
        return symmetric(root.left,root.right);
    }
    public boolean symmetric(TreeNode left , TreeNode right) {
        if(left == null && right == null )
            return true;
        if(right == null || left == null )
            return false;
        return left.val == right.val && symmetric(left.left , right.right)&&symmetric(left.right , right.left);
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        int a = s1.length();
        int b = s2.length();
        int c = s3.length();
        if(c != a+b)
            return false;
        if(a == 0)
            return s2.equals(s3);
        if(b == 0)
            return s1.equals(s3);
        boolean[][] dp = new boolean[a+1][b+1];
        dp[0][0] = true;
        for(int i = 1 ; i<= a;i++)
            dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
        for(int j = 1 ; j<= b; j++)
            dp[0][j] = dp[0][j-1] && s2.charAt(j-1) == s3.charAt(j-1);
        for(int i = 1 ; i<=a;i++)
            for(int j = 1 ; j<=b ; j++) {
                dp[i][j] = dp[i-1][j]&&s1.charAt(i-1) == s3.charAt(i+j-1) || dp[i][j-1] &&s2.charAt(j-1) == s3.charAt(i+j-1);
            }
        return dp[a][b];
    }

    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        if(root == null)
            return list;
        TreeNode pre = root;

        while(!stack.empty() ||pre!=null) {
            while(pre!=null) {
                stack.push(pre);
                pre = pre.left;
            }
            if(!stack.empty()) {
                TreeNode node = stack.pop();
                list.add(node.val);
                pre = node.right;
            }
        }
        return list;
    }

    private ArrayList<String> list = new ArrayList<>();
    public ArrayList<String> restoreIpAddresses(String s) {
        ArrayList<String> ip = new ArrayList<>();
        dfs(s,ip,0);
        return list;
    }
    public void dfs(String s, ArrayList<String> ip , int start) {
        if(ip.size() == 4 && start == s.length()) {
            list.add(ip.get(0)+'.'+ip.get(1)+'.'+ip.get(2)+'.'+ip.get(3));
            return ;
        }
        if(s.length()-start>3*(4-ip.size()) || s.length()-start<4-ip.size())
            return;
        int num = 0;
        for(int i = start ; i<start+3&&i<s.length() ; i++) {
            num = num*10 + s.charAt(i)-'0';
            if(num< 0 || num>255)
                return ;
            ip.add(s.substring(start , i+1));
            dfs(s , ip , i+1);
            ip.remove(ip.size()-1);
            if(num == 0)
                break;
        }
    }

    public ArrayList<ArrayList<Integer>> Alllist = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
        Arrays.sort(num);
        ArrayList<Integer> list = new ArrayList<>();
        findsubSet(num , 0 , list);
        return Alllist;

    }
    public void findsubSet(int[] num , int start , ArrayList<Integer> list ) {
        Alllist.add(new ArrayList<>(list));
        for(int i = start ; i<num.length;i++) {
            if(i>start && num[i] == num[i-1])
                continue;
            list.add(num[i]);
            findsubSet(num,i+1,list);
            list.remove(list.size()-1);
        }
    }

    public int numDecodings(String s) {
        if(s == null)
            return 0;
        if(s.equals("")||s.charAt(0) == '0')
            return 0;
        for(int i = 1;i<s.length();i++) {
            if(s.charAt(i) == '0' && (s.charAt(i-1) =='0' || s.charAt(i-1) - '0'>2))
                return 0;
        }

        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 1 ; i<s.length() ; i++) {
            if(s.charAt(i) != '0')
                dp[i+1] += dp[i];
            if((s.charAt(i-1) - '0')*10+(s.charAt(i) - '0')<=26)
                dp[i+1]+=dp[i-1];
        }
        return dp[s.length()];
    }

    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2))
            return true;
        if(s1.length()!=s2.length())
            return false;
        int[] letters = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letters[s1.charAt(i) - 'a']++;
            letters[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++)
            if (letters[i] != 0)
            return false;
        for(int i = 1; i<s1.length(); i++) {
            if(isScramble(s1.substring(0,i),s2.substring(0,i)) && isScramble(s1.substring(i),s2.substring(i)) ||
                    isScramble(s1.substring(0,i),s2.substring(i)) && isScramble(s1.substring(i),s2.substring(0,i)))
                return true;
        }
        return false;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode vHead = new ListNode(0);
        vHead.next = head;
        ListNode in = vHead;
        ListNode p = head;
        ListNode pre = vHead;
        while(p!=null && p.val<x) {
            pre = p;
            in = p;
            p = p.next;
        }
        while(p != null) {
            if(p.val<x) {
                ListNode node = p;
                p = p.next;
                pre.next = p;
                node.next = in.next;
                in.next = node;
                in = in.next;
            } else {
                pre = p;
                p = p.next;
            }
        }
        return vHead.next;
    }

    public int largestRectangleArea(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int max = 0;
        for(int i = 0; i<=height.length; i++) {
            int h = (i == height.length ? 0 :height[i]);
            if(stack.isEmpty() || h>= height[stack.peek()]) {
                stack.push(i);
            } else {
                int p = stack.pop();
                max = Math.max(max , height[p]*(stack.isEmpty()? i : i-1-stack.peek() ));
                i--;
            }
        }
        return max;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if(head == null ||head.next == null)
            return head;
        ListNode vHead = new ListNode(0);
        vHead.next = head;
        ListNode pre = vHead;
        ListNode p = head;
        while(p!=null) {
            boolean flag = false;
            while(p.next!=null &&p.val == p.next.val) {
                p = p.next;
                flag = true;
            }
            if(flag) {
                pre.next = p;
            } else {
                pre.next = p;
                pre = p;
            }
            p = p.next;
        }
        return vHead.next;
    }

    public int removeDuplicates(int[] A) {
        int len = A.length;
        if(len<=2)
            return len;
        int c =A[0]-1;
        int i = 1;
        while(i < A.length ) {
            int num = 1;
            while(i<A.length && A[i] == A[i-1]) {
                num++;
                i++;
            }
            if(num>2) {
                int dup = num - 2;
                for(int m = 1 ; m <= dup ; m++){
                    A[i-m] = c;
                }
                len -= dup;
            }
            i++;
        }
        int j = 2;
        for(i = 2; i<len; i++ ) {
            if(A[j]!= c) {
                A[i] = A[j];
                j++;
            } else {
                j++;
                i--;
            }
        }
        return len;
    }

    public boolean exist(char[][] board, String word) {
        boolean[][] flag = new boolean[board.length][board[0].length];
        if(board == null)
            return false;
        for(int i = 0 ;i<board.length ; i++)
            for(int j = 0 ; j < board[0].length; j++) {
                if (dfs(board, flag, word, i, j))
                    return true;
            }
        return false;
    }
    public boolean dfs(char[][] board,boolean[][] flag, String word, int i, int j) {
        if(i<0 || i>=board.length || j<0 || j>=board[0].length || flag[i][j] || board[i][j] != word.charAt(0))
            return false;
        if(word.length() == 1)
            return true;
        flag[i][j] = true;
        boolean ret = dfs(board, flag, word.substring(1), i-1, j ) || dfs(board, flag, word.substring(1), i+1, j )
                || dfs(board, flag, word.substring(1), i, j+1 ) || dfs(board, flag, word.substring(1), i, j-1 );
        if(ret)
            return true;
        else {
            flag[i][j] = false;
            return false;
        }

    }

    public int maximalRectangle(char[][] matrix) {
        int area = 0;
        for(int i = 0 ; i<matrix.length;i++)
            for(int j = 0 ; j<matrix[0].length;j++) {
                if(matrix[i][j] == '1') {
                    int m = i;
                    int n = j;
                    while(n<matrix[0].length) {
                        if(matrix[m][n] == '1') {
                            n++;
                            area = Math.max(area , n-j);
                        } else
                            break;
                    }
                    m++;
                    int board = n;
                    int index = n;
                    while(m<matrix.length && matrix[m][j] == '1') {
                        board = index;
                        index = j;
                        while(index<board && matrix[m][index] == '1') {
                            area = Math.max(area , (m-i+1)*(index-j+1));
                            index++;
                        }
                        m++;
                    }
                }
            }
        return area;
    }

    ArrayList<ArrayList<Integer>> LIST = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        ArrayList<Integer> list = new ArrayList<>();
        dfs(S, list,0);
        return LIST;

    }
    public void dfs(int[] S, ArrayList<Integer> list , int index) {
        LIST.add(new ArrayList<Integer>(list));
        for(int i = index; i<S.length; i++) {
            list.add(S[i]);
            dfs(S, list, i+1);
            list.remove(list.size()-1);
        }
    }

    public String minWindow(String S, String T) {
    int[] map = new int[128];
    //init map, 记录T中每个元素出现的次数
    for(int i = 0; i < T.length(); i++) {
        map[T.charAt(i)]++;
    }

    // begin end两个指针指向窗口的首位，d记录窗口的长度， counter记录T中还有几个字符没被窗口包含
    int begin = 0, end = 0, d = Integer.MAX_VALUE, counter = T.length(), head = 0;
    // end指针一直向后遍历
    while(end < S.length()) {
    // map[] > 0 说明该字符在T中出现，counter-- 表示对应的字符被包含在了窗口，counter--, 如果s中的字符没有在T中出现，则map[]中对应的字符-1后变为负值
        if(map[S.charAt(end++)]-- > 0) {
        counter--;
        }
    // 当counter==0时，说明窗口已经包含了T中的所有字符
        while (counter == 0) {
            if(end - begin < d) {
            d = end - (head = begin);
            }
            if(map[S.charAt(begin++)]++ == 0) {// begin开始后移，继续向后寻找。如果begin后移后指向的字符在map中==0，表示是在T中出现的，如果没有出现，map[]中的值会是负值。
            counter++;// 在T中的某个字符从窗口中移除，所以counter++。
            }
        }
    }
    return d==Integer.MAX_VALUE ? "" :S.substring(head, head+d);
}

    public void sortColors(int[] A) {
        int begin = 0, end = A.length;
        for (int i = 0; i < A.length; i++) {
            if (A[i] == 0) {
                swap(A, i, begin);
                begin++;
            }
            else if(A[i]== 2) {
                swap(A, i, end);
                end--;
                i--;
            }
        }
    }
    public void swap(int[] A , int i, int j) {
        int swap = A[i];
        A[i] = A[j];
        A[j] = swap;
    }

    public void quickmerge(int[] A, int left, int right) {
        if(left>=right)
            return ;
        int index = quicksort(A, left, right);
        quickmerge(A,left,index-1);
        quickmerge(A,index+1,right);
    }
    public int quicksort(int[] A, int i, int j) {
        int div = A[i];
        while(i!=j) {
            while(j>i &&A[j]>= div) {
                j--;
            }
            if(j>i)
                A[i] = A[j];
            while(i<j&&A[i]< div) {
                i++;
            }
            if(i<j)
                A[j] = A[i];
        }
        A[i] = div;
        return i;
    }

    public void setZeroes(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        boolean flag_row = false, flag_col = false;
        for(int i = 0; i<rows; i++) {
            if(matrix[i][0] == 0)
                flag_col = true;
        }
        for(int i = 0; i<cols; i++) {
            if(matrix[0][i] == 0)
                flag_row = true;
        }
        for(int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        for(int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }

        if(flag_col) {
            for(int i = 0; i<rows; i++)
                matrix[i][0] = 0;
        }
        if(flag_row) {
            for(int i = 0; i<cols; i++)
                matrix[0][i] = 0;
        }
        return ;
    }

    public int minDistance(String word1, String word2) {
        if(word1 == null && word2 == null)
            return 0;
        if(word1 == null || word1.length() == 0)
            return word2.length();
        if(word2 == null || word2.length() == 0)
            return word1.length();
        int[][] dp = new int[word1.length()+1][word2.length()+1];
        dp[0][0] = 0;
        for(int i = 1;i<=word1.length();i++)
            dp[i][0] = i;
        for(int i = 1; i <=word2.length(); i++)
            dp[0][i] = i;
        for(int j = 1; j <=word2.length(); j++) {
            for(int i = 1; i<=word1.length(); i++) {
                dp[i][j] = Math.min(dp[i-1][j]+1,dp[i-1][j-1]+((word1.charAt(i-1) == word2.charAt(j-1))? 0 : 1));
                dp[i][j] = Math.min(dp[i][j],dp[i][j-1]+1);
            }
        }
        return dp[word1.length()][word2.length()];
    }
    public static void main(String[] args) {
        Solution test = new Solution();
        int a = test.minDistance("a","ab");
        System.out.println(a);
        System.out.println(" branch dev");
    }
}

