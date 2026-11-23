import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Set.Countable
import Mathlib.Data.Multiset.MapFold
import Mathlib.Data.Finset.Card
import Lean

open Lean Elab Command

def mathlibSourceRevision : String :=
  "6ebcce37335bcb6bca64596e5d57bb3035648030"

def leanSourceRevision : String :=
  "0b77e12952ad39077d80d8bd6b66d601f225c321"

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

  let output := Json.mkObj
    [ ("schema_version", toJson "1.0.0")
    , ("source_repository", toJson "leanprover-community/mathlib4")
    , ("source_revision", toJson mathlibSourceRevision)
    , ("lean_toolchain", toJson "leanprover/lean4:v4.37.0-rc1")
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
    ]

  let outputPath := s!"proof-comprehension-extract-{name}.json"
  liftIO <| IO.FS.writeFile outputPath (output.compress ++ "\n")
  logInfo m!"PROOF_COMPREHENSION_EXTRACTED declaration={name} path={outputPath}"

#emit_proof_comprehension Nat.choose_pos
#emit_proof_comprehension Set.countable_union
#emit_proof_comprehension Multiset.map_id
#emit_proof_comprehension Finset.card_map
