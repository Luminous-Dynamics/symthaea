package independence

import (
	"fmt"
	"strings"
	"testing"
)

func largeInput(chain int) Input {
	in:=fixture()
	for _,observer:=range in.Observers {
		leaf:=make([]string,0,len(criticalDomainClasses))
		prefix:=string([]rune(observer.ID)[len([]rune("observer-")):])
		for _,node:=range in.Nodes {
			if node.FailureClass!="" && strings.HasPrefix(node.ID,prefix+"-") { leaf=append(leaf,node.ID) }
		}
		last:=observer.ID+"-chain-0"
		in.Nodes=append(in.Nodes,Node{ID:last,DependsOn:leaf})
		for i:=1;i<chain;i++ {
			next:=fmt.Sprintf("%s-chain-%d",observer.ID,i)
			in.Nodes=append(in.Nodes,Node{ID:next,DependsOn:[]string{last}})
			last=next
		}
		for i:=range in.Nodes {
			if in.Nodes[i].ID==observer.RootNodeID { in.Nodes[i].DependsOn=[]string{last} }
		}
	}
	return in
}

func BenchmarkEvaluateTwoObserverClosures1000(b *testing.B) {
	in:=largeInput(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i:=0;i<b.N;i++ {
		result:=Evaluate(in)
		if result.Verdict!=Qualified { b.Fatalf("expected independent pair: %+v",result) }
	}
}
