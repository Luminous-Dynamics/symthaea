dtmc

// Two-state, three-reward-layer interval DTMC.
// The clock makes the uncertain transition rows stage-specific. The extra
// terminal clock value has zero reward so total-reward queries are finite.
module interval_markov

    s : [0..1] init 0;
    t : [0..3] init 0;

    // Stage 0.
    [] t=0 & s=0 ->
        [0.4,0.8] : (s'=0) & (t'=1)
      + [0.2,0.6] : (s'=1) & (t'=1);

    [] t=0 & s=1 ->
        [0.3,0.7] : (s'=0) & (t'=1)
      + [0.3,0.7] : (s'=1) & (t'=1);

    // Stage 1.
    [] t=1 & s=0 ->
        [0.5,0.9] : (s'=0) & (t'=2)
      + [0.1,0.5] : (s'=1) & (t'=2);

    [] t=1 & s=1 ->
        [0.2,0.6] : (s'=0) & (t'=2)
      + [0.4,0.8] : (s'=1) & (t'=2);

    // Zero-reward terminal transition and absorbing state.
    [] t=2 -> 1 : (t'=3);
    [] t=3 -> 1 : (t'=3);

endmodule

rewards "action_a"
    t=1 & s=1 : 1;
    t=2 & s=1 : 1;
endrewards

rewards "action_b"
    t=1 & s=0 : 1;
endrewards

rewards "action_c"
    t=1 : 0.5;
    t=2 & s=1 : 0.5;
endrewards

// Every ordered pair compares score functions under the same transition
// process. Each shifted reward adds a path-constant offset, which is removed
// from the resulting maximum total reward.
rewards "regret_a_b_shifted"
    t=1 & s=0 : 2;
    t=2 & s=0 : 1;
endrewards

rewards "regret_a_c_shifted"
    t=1 & s=0 : 1;
    t=2 & s=0 : 0.5;
endrewards

rewards "regret_b_a_shifted"
    t=1 & s=1 : 2;
    t=2 & s=1 : 1;
endrewards

rewards "regret_b_c_shifted"
    t=1 & s=1 : 1;
    t=2 & s=1 : 0.5;
endrewards

rewards "regret_c_a_shifted"
    t=1 & s=1 : 1;
    t=2 & s=1 : 0.5;
endrewards

rewards "regret_c_b_shifted"
    t=1 & s=0 : 1;
    t=2 & s=0 : 0.5;
endrewards
