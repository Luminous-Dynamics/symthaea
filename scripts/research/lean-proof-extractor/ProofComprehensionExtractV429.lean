import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Set.Countable
import Mathlib.Data.Multiset.MapFold
import Mathlib.Data.Finset.Card
import Lean

open Lean Elab Command

def mathlibSourceRevision : String :=
  "8a178386ffc0f5fef0b77738bb5449d50efeea95"

def leanSourceRevision : String :=
  "98dc76e3c0a9b856c9b98726b713fb04fab16740"

def constantKind : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "definition"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quotient"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

def namesJson (names : Array Name) : Json :=
  Json.arr <| names.map fun name => toJson (toString name)

def edgeJson (source relation origin : String) (target : Name) : Json :=
  Json.mkObj
    [ ("source", toJson source)
    , ("target", toJson (toString target))
    , ("relation", toJson relation)
    , ("origin", toJson origin)
    ]

def nameEdges (source relation origin : String) (targets : Array Name) : Array Json :=
  targets.map fun target => edgeJson source relation origin target

def expressionEdges (source relation origin : String) (expression : Expr) : Array Json :=
  nameEdges source relation origin expression.getUsedConstants

def structuralEdges (source : String) (info : ConstantInfo) : Array Json :=
  match info with
  | .inductInfo inductiveInfo =>
      nameEdges source "inductive_constructor" "ConstantInfo.inductInfo.ctors"
        inductiveInfo.ctors.toArray
      ++ nameEdges source "mutual_inductive_member" "ConstantInfo.inductInfo.all"
        inductiveInfo.all.toArray
  | .ctorInfo constructorInfo =>
      #[edgeJson source "constructor_of" "ConstantInfo.ctorInfo.induct"
        constructorInfo.induct]
  | .recInfo recursorInfo =>
      recursorInfo.rules.foldl
        (fun edges rule =>
          edges.push
            (edgeJson source "recursor_rule_constructor"
              "ConstantInfo.recInfo.rules.ctor" rule.ctor)
          ++ expressionEdges source "recursor_reduction_dependency"
              "ConstantInfo.recInfo.rules.rhs" rule.rhs)
        (nameEdges source "recursor_for_inductive" "ConstantInfo.recInfo.all"
          recursorInfo.all.toArray)
  | _ => #[]


def startsWithNameParts : List Name → List Name → Bool
  | [], _ => true
  | _ :: _, [] => false
  | expected :: expectedTail, actual :: actualTail =>
      expected == actual && startsWithNameParts expectedTail actualTail

partial def containsNameParts (needle haystack : List Name) : Bool :=
  match haystack with
  | [] => false
  | _ :: rest => startsWithNameParts needle haystack || containsNameParts needle rest

def isGeneratedHelperFor (root candidate : Name) : Bool :=
  candidate != root
    && candidate.components.length > root.components.length
    && containsNameParts root.components candidate.components

elab "#emit_proof_comprehension " declaration:ident : command => do
  let name := declaration.getId
  let env ← getEnv
  let some info := env.find? name
    | throwError "proof-comprehension extractor: declaration {name} not found"

  let typeDependencies := info.type.getUsedConstantsAsSet.toList.toArray
  let value? := info.value? (allowOpaque := true)
  let proofValueDependencies := value?.map Expr.getUsedConstants |>.getD #[]
  let closureDependencies := info.getUsedConstantsAsSet.toList.toArray
  let unresolvedDependencies :=
    closureDependencies.filter fun dependency => (env.find? dependency).isNone
  let declarationName := toString name

  let edges :=
    nameEdges declarationName "type_dependency" "ConstantInfo.type"
      typeDependencies
    ++ nameEdges declarationName "proof_dependency" "ConstantInfo.value?"
      proofValueDependencies
    ++ structuralEdges declarationName info

  let mut helperNodes : Array Json := #[]
  for helperName in proofValueDependencies do
    if isGeneratedHelperFor name helperName then
      let some helperInfo := env.find? helperName
        | throwError "proof-comprehension helper {helperName} referenced by {name} is not in the environment"
      let helperTypeDependencies := helperInfo.type.getUsedConstantsAsSet.toList.toArray
      let helperValue? := helperInfo.value? (allowOpaque := true)
      let helperProofDependencies := helperValue?.map Expr.getUsedConstants |>.getD #[]
      let helperClosure := helperInfo.getUsedConstantsAsSet.toList.toArray
      let helperUnresolved :=
        helperClosure.filter fun dependency => (env.find? dependency).isNone
      let helperEdges :=
        nameEdges (toString helperName) "type_dependency" "ConstantInfo.type"
          helperTypeDependencies
        ++ nameEdges (toString helperName) "proof_dependency" "ConstantInfo.value?"
          helperProofDependencies
        ++ structuralEdges (toString helperName) helperInfo
      helperNodes := helperNodes.push <| Json.mkObj
        [ ("root_declaration", toJson declarationName)
        , ("node_role", toJson "generated_or_private_helper")
        , ("declaration", toJson (toString helperName))
        , ("declaration_kind", toJson (constantKind helperInfo))
        , ("proof_value_available", toJson helperValue?.isSome)
        , ("type_dependencies", namesJson helperTypeDependencies)
        , ("proof_value_dependencies", namesJson helperProofDependencies)
        , ("constant_info_dependency_closure", namesJson helperClosure)
        , ("unresolved_dependencies", namesJson helperUnresolved)
        , ("edges", Json.arr helperEdges)
        ]

  let output := Json.mkObj
    [ ("schema_version", toJson "1.0.0")
    , ("source_repository", toJson "leanprover-community/mathlib4")
    , ("source_revision", toJson mathlibSourceRevision)
    , ("lean_toolchain", toJson "leanprover/lean4:v4.29.0")
    , ("lean_revision", toJson leanSourceRevision)
    , ("declaration", toJson declarationName)
    , ("declaration_kind", toJson (constantKind info))
    , ("level_parameters", namesJson info.levelParams.toArray)
    , ("proof_value_available", toJson value?.isSome)
    , ("type_dependencies", namesJson typeDependencies)
    , ("proof_value_dependencies", namesJson proofValueDependencies)
    , ("constant_info_dependency_closure", namesJson closureDependencies)
    , ("unresolved_dependencies", namesJson unresolvedDependencies)
    , ("edges", Json.arr edges)
    , ("helper_nodes", Json.arr helperNodes)
    ]

  let outputPath := s!"proof-comprehension-extract-{name}.json"
  liftIO <| IO.FS.writeFile outputPath (output.compress ++ "\n")
  logInfo m!"PROOF_COMPREHENSION_EXTRACTED declaration={name} path={outputPath}"

#emit_proof_comprehension Nat.choose_pos
#emit_proof_comprehension Set.countable_union
#emit_proof_comprehension Multiset.map_id
#emit_proof_comprehension Finset.card_map
