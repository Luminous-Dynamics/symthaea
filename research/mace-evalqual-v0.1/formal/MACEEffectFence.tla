---------------------- MODULE MACEEffectFence ----------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    K1, K2,
    I1, I2,
    NoKey, NoIntent,
    NoTerminal, Effect, Fence,
    NoTxKind, EffectTx, FenceTx,
    Found, NotExecutedFenced, NotFoundUnfenced

Keys == {K1, K2}
Intents == {I1, I2}
Terminals == {NoTerminal, Effect, Fence}
TxKinds == {NoTxKind, EffectTx, FenceTx}
PendingRequests == {<<k, i>> : k \in Keys, i \in Intents}

ASSUME
    /\ K1 # K2
    /\ I1 # I2
    /\ NoKey \notin Keys
    /\ NoIntent \notin Intents
    /\ NoTerminal \notin {Effect, Fence}
    /\ Effect # Fence
    /\ NoTxKind \notin {EffectTx, FenceTx}
    /\ EffectTx # FenceTx
    /\ Found # NotExecutedFenced
    /\ Found # NotFoundUnfenced
    /\ NotExecutedFenced # NotFoundUnfenced

VARIABLES
    terminal,
    terminalIntent,
    receiptVisible,
    businessVisible,
    pending,
    txKey,
    txIntent,
    txKind,
    txReceipt,
    txBusiness

vars ==
    << terminal, terminalIntent, receiptVisible, businessVisible,
       pending, txKey, txIntent, txKind, txReceipt, txBusiness >>

Init ==
    /\ terminal = [k \in Keys |-> NoTerminal]
    /\ terminalIntent = [k \in Keys |-> NoIntent]
    /\ receiptVisible = [k \in Keys |-> FALSE]
    /\ businessVisible = [k \in Keys |-> FALSE]
    /\ pending = {}
    /\ txKey = NoKey
    /\ txIntent = NoIntent
    /\ txKind = NoTxKind
    /\ txReceipt = FALSE
    /\ txBusiness = FALSE

Send(k, i) ==
    /\ k \in Keys
    /\ i \in Intents
    /\ <<k, i>> \notin pending
    /\ pending' = pending \cup {<<k, i>>}
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible,
           txKey, txIntent, txKind, txReceipt, txBusiness >>

BeginEffect(k, i) ==
    /\ txKind = NoTxKind
    /\ k \in Keys
    /\ i \in Intents
    /\ <<k, i>> \in pending
    /\ pending' = pending \ {<<k, i>>}
    /\ IF terminal[k] = NoTerminal
          THEN
            /\ txKey' = k
            /\ txIntent' = i
            /\ txKind' = EffectTx
            /\ txReceipt' = FALSE
            /\ txBusiness' = FALSE
          ELSE
            UNCHANGED << txKey, txIntent, txKind, txReceipt, txBusiness >>
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible >>

BeginFence(k, i) ==
    /\ txKind = NoTxKind
    /\ k \in Keys
    /\ i \in Intents
    /\ IF terminal[k] = NoTerminal
          THEN
            /\ txKey' = k
            /\ txIntent' = i
            /\ txKind' = FenceTx
            /\ txReceipt' = FALSE
            /\ txBusiness' = FALSE
          ELSE
            UNCHANGED << txKey, txIntent, txKind, txReceipt, txBusiness >>
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible, pending >>

StageReceipt ==
    /\ txKind # NoTxKind
    /\ txReceipt = FALSE
    /\ txReceipt' = TRUE
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible,
           pending, txKey, txIntent, txKind, txBusiness >>

StageBusiness ==
    /\ txKind = EffectTx
    /\ txBusiness = FALSE
    /\ txBusiness' = TRUE
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible,
           pending, txKey, txIntent, txKind, txReceipt >>

CommitEffect ==
    /\ txKind = EffectTx
    /\ txReceipt = TRUE
    /\ txBusiness = TRUE
    /\ terminal[txKey] = NoTerminal
    /\ terminal' = [terminal EXCEPT ![txKey] = Effect]
    /\ terminalIntent' = [terminalIntent EXCEPT ![txKey] = txIntent]
    /\ receiptVisible' = [receiptVisible EXCEPT ![txKey] = TRUE]
    /\ businessVisible' = [businessVisible EXCEPT ![txKey] = TRUE]
    /\ txKey' = NoKey
    /\ txIntent' = NoIntent
    /\ txKind' = NoTxKind
    /\ txReceipt' = FALSE
    /\ txBusiness' = FALSE
    /\ UNCHANGED pending

CommitFence ==
    /\ txKind = FenceTx
    /\ txReceipt = TRUE
    /\ txBusiness = FALSE
    /\ terminal[txKey] = NoTerminal
    /\ terminal' = [terminal EXCEPT ![txKey] = Fence]
    /\ terminalIntent' = [terminalIntent EXCEPT ![txKey] = txIntent]
    /\ receiptVisible' = [receiptVisible EXCEPT ![txKey] = TRUE]
    /\ txKey' = NoKey
    /\ txIntent' = NoIntent
    /\ txKind' = NoTxKind
    /\ txReceipt' = FALSE
    /\ txBusiness' = FALSE
    /\ UNCHANGED << businessVisible, pending >>

RollbackOrCrash ==
    /\ txKind # NoTxKind
    /\ txKey' = NoKey
    /\ txIntent' = NoIntent
    /\ txKind' = NoTxKind
    /\ txReceipt' = FALSE
    /\ txBusiness' = FALSE
    /\ UNCHANGED
        << terminal, terminalIntent, receiptVisible, businessVisible, pending >>

Reconcile(k) ==
    IF terminal[k] = Effect
       THEN Found
       ELSE IF terminal[k] = Fence
               THEN NotExecutedFenced
               ELSE NotFoundUnfenced

Next ==
    \/ \E k \in Keys, i \in Intents : Send(k, i)
    \/ \E k \in Keys, i \in Intents : BeginEffect(k, i)
    \/ \E k \in Keys, i \in Intents : BeginFence(k, i)
    \/ StageReceipt
    \/ StageBusiness
    \/ CommitEffect
    \/ CommitFence
    \/ RollbackOrCrash

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ terminal \in [Keys -> Terminals]
    /\ terminalIntent \in [Keys -> (Intents \cup {NoIntent})]
    /\ receiptVisible \in [Keys -> BOOLEAN]
    /\ businessVisible \in [Keys -> BOOLEAN]
    /\ pending \subseteq PendingRequests
    /\ txKey \in (Keys \cup {NoKey})
    /\ txIntent \in (Intents \cup {NoIntent})
    /\ txKind \in TxKinds
    /\ txReceipt \in BOOLEAN
    /\ txBusiness \in BOOLEAN

TerminalVisibility ==
    \A k \in Keys :
        /\ (terminal[k] = NoTerminal
              => /\ receiptVisible[k] = FALSE
                 /\ businessVisible[k] = FALSE)
        /\ (terminal[k] = Effect
              => /\ receiptVisible[k] = TRUE
                 /\ businessVisible[k] = TRUE)
        /\ (terminal[k] = Fence
              => /\ receiptVisible[k] = TRUE
                 /\ businessVisible[k] = FALSE)

TransactionWellFormed ==
    /\ (txKind = NoTxKind
          => /\ txKey = NoKey
             /\ txIntent = NoIntent
             /\ txReceipt = FALSE
             /\ txBusiness = FALSE)
    /\ (txKind # NoTxKind
          => /\ txKey \in Keys
             /\ txIntent \in Intents
             /\ terminal[txKey] = NoTerminal
             /\ (txKind = FenceTx => txBusiness = FALSE))

CommittedIntentDefined ==
    \A k \in Keys :
        (terminal[k] = NoTerminal => terminalIntent[k] = NoIntent)
        /\ (terminal[k] # NoTerminal => terminalIntent[k] \in Intents)

ReconciliationSound ==
    \A k \in Keys :
        /\ (terminal[k] = NoTerminal => Reconcile(k) = NotFoundUnfenced)
        /\ (terminal[k] = Effect => Reconcile(k) = Found)
        /\ (terminal[k] = Fence => Reconcile(k) = NotExecutedFenced)

FenceNeverExecutes ==
    \A k \in Keys :
        terminal[k] = Fence => businessVisible[k] = FALSE

=============================================================================
