# ENG-REA-005 canonical semantic projection of Unblob 26.6.4 ProcessResult JSON.
# --arg root ABSOLUTE_SCRATCH_ROOT names a disposable scratch root containing input/output paths.
# Raw provider JSON bytes must be hashed and retained separately; this file never rewrites them.
# Fail closed on unfamiliar reports, escaping paths, malformed tasks and ambiguous lineage.

def uint($n): ($n|type)=="number" and $n>=0 and ($n|floor)==$n;
def stablejson: walk(if type=="object" then (to_entries|sort_by(.key)|from_entries) else . end)|tojson;

def known_report_types: [
  "UnknownError", "CalculateChunkExceptionReport", "CalculateMultiFileExceptionReport",
  "ExtractCommandFailedReport", "OutputDirectoryExistsReport", "ExtractorDependencyNotFoundReport",
  "ExtractedFileDeletedReport", "ExtractorTimedOut", "MaliciousSymlinkRemoved",
  "MultiFileCollisionReport", "StatReport", "HashReport", "FileMagicReport",
  "RandomnessReport", "ChunkReport", "UnknownChunkReport",
  "CarveDirectoryReport", "MultiFileReport", "ExtractionProblem", "PathTraversalProblem",
  "LinkExtractionProblem", "SpecialFileExtractionProblem", "ExtendedAttributeExtractionProblem"
];

def relpath($root; $p):
  if ($p|type)!="string" or ($p|length)==0 then error("path field is missing or not a string")
  elif $p==$root then "."
  elif ($p|startswith($root + "/")) then $p[(($root|length)+1):]
  elif ($p|startswith("/")) then error("absolute path escapes declared scratch root: " + $p)
  elif (($p|split("/"))|index("..")) != null then error("relative path traverses parent: " + $p)
  else $p end;

def normalize_paths($root):
  walk(
    if type=="object" then
      with_entries(
        . as $entry |
        if (["path", "carve_dir", "extraction_path", "link_path"]|index($entry.key)) != null
           and ($entry.value|type)=="string" then
          .value = relpath($root; $entry.value)
        elif $entry.key=="paths" and ($entry.value|type)=="array" then
          .value |= map(relpath($root; .))
        else . end
      )
    else . end
  );

def canonical_report($root):
  if type!="object" then error("report record is not an object")
  elif (.__typename__|type)!="string" then error("report missing __typename__")
  elif (.__typename__ as $t | known_report_types | index($t))==null then error("unknown report type: " + .__typename__)
  else
    (if (. as $r | (["ChunkReport", "UnknownChunkReport", "MultiFileReport"]|index($r.__typename__))) != null
      then del(.id) else . end)
    | if has("extraction_reports") then
        if (.extraction_reports|type)!="array" then error("extraction_reports must be an array")
        else .extraction_reports |= (map(canonical_report($root))|sort_by(stablejson)) end
      else . end
    | if has("metadata_reports") then
        if (.metadata_reports|type)!="array" then error("metadata_reports must be an array")
        else .metadata_reports |= (map(canonical_report($root))|sort_by(stablejson)) end
      else . end
    | normalize_paths($root)
  end;

def task_fields_ok($t):
  ($t|type)=="object" and
  (($t|keys|sort)==["blob_id", "depth", "is_multi_file", "path"]) and
  ($t.path|type)=="string" and ($t.path|length)>0 and
  uint($t.depth) and ($t.blob_id|type)=="string" and ($t.is_multi_file|type)=="boolean";

def bad_task_result:
  if type!="object" then true
  elif ((keys|sort)!=["reports", "subtasks", "task"]) then true
  elif (task_fields_ok(.task)|not) then true
  elif (.reports|type)!="array" then true
  elif (.subtasks|type)!="array" then true
  elif any(.subtasks[]; task_fields_ok(.)|not) then true
  else false end;

def bad_chunk($task; $chunk):
  if ($chunk.id|type)!="string" or ($chunk.id|length)==0 then true
  elif (uint($chunk.start_offset)|not) or (uint($chunk.end_offset)|not) or (uint($chunk.size)|not) then true
  elif $chunk.end_offset <= $chunk.start_offset then true
  elif $chunk.size != ($chunk.end_offset-$chunk.start_offset) then true
  else
    ([$task.reports[]? | select(.__typename__=="StatReport" and .is_file==true) | .size] | first) as $task_size
    | if $task_size==null then false
      elif (uint($task_size)|not) then true
      else $chunk.end_offset > $task_size end
  end;

def producer_key($chunkmap; $owner; $id):
  ($chunkmap|map(select(.id==$id and .owner==$owner))) as $matches
  | if ($matches|length)!=1 then error("missing or ambiguous producer on parent task: " + $id)
    else $matches[0].key end;

def producer_key_global($chunkmap; $id):
  ($chunkmap|map(select(.id==$id))) as $matches
  | if ($matches|length)!=1 then error("missing or ambiguous producer reference: " + $id)
    else $matches[0].key end;

def has_directory_stat($tr):
  any($tr.reports[]?; .__typename__=="StatReport" and .is_dir==true and .is_file==false);

def edge_kind($parent; $sub):
  if $sub.depth==$parent.task.depth+1 then "producer-child"
  elif $sub.depth==$parent.task.depth and $sub.blob_id==$parent.task.blob_id then "directory-expansion"
  else "invalid" end;

def bad_edge($edge; $all; $root; $chunkmap):
  ([$all[]|select(.task.path==$edge.parent)] ) as $parents
  | ([$all[]|select(.task.path==$edge.child)] ) as $children
  | if ($parents|length)!=1 or ($children|length)!=1 then true
    elif $edge.parent==$edge.child then true
    elif (task_fields_ok($edge.subtask)|not) then true
    elif $edge.subtask.path!=$children[0].task.path or
         $edge.subtask.blob_id!=$children[0].task.blob_id or
         $edge.subtask.depth!=$children[0].task.depth or
         $edge.subtask.is_multi_file!=$children[0].task.is_multi_file then true
    elif $edge.depth!=$children[0].task.depth then true
    elif edge_kind($parents[0]; $edge.subtask)=="producer-child" then
      ([$parents[0].reports[]?|select((.__typename__=="ChunkReport" or .__typename__=="MultiFileReport") and .id==$edge.subtask.blob_id)]|length)!=1
    elif edge_kind($parents[0]; $edge.subtask)=="directory-expansion" then
      ($edge.subtask.blob_id!=$parents[0].task.blob_id or (has_directory_stat($parents[0])|not))
    else true end;

def project($root):
  . as $all
  | if (($root|type)!="string") or ((($root|startswith("/"))|not)) or ($root|endswith("/")) then
      error("root must be an absolute scratch path without a trailing slash")
    elif ($all|type)!="array" or ($all|length)==0 then
      error("provider report must be a nonempty TaskResult array")
    elif any($all[]; bad_task_result) then
      error("malformed Unblob TaskResult / Task / subtask schema")
    else
      [$all[] as $tr | {raw:$tr, path:relpath($root; $tr.task.path)}] as $tasks
      | if (($tasks|map(.path)|unique|length)!=($tasks|length)) then
          error("duplicate normalized task path")
        else
          [ $all[] as $tr
            | $tr.reports[]?
            | select(.__typename__=="ChunkReport" or .__typename__=="MultiFileReport")
            | {id:.id, owner:relpath($root; $tr.task.path),
               key:{task_path:relpath($root; $tr.task.path), chunk:canonical_report($root)}}
          ] as $chunkmap
          | if any($all[]; . as $tr | any($tr.reports[]?; ((.__typename__=="ChunkReport" or .__typename__=="UnknownChunkReport") and (bad_chunk($tr; .))))) then
              error("malformed chunk interval or out-of-bounds chunk")
            elif (($chunkmap|map(.id)|unique|length)!=($chunkmap|length)) then
              error("duplicate producer ID")
            elif any($chunkmap[]; (.id|type)!="string" or (.id|length)==0) then
              error("empty producer ID")
            else
              [ $all[] as $parent | $parent.subtasks[]? as $sub
                | {parent:$parent.task.path, child:$sub.path, subtask:$sub, depth:$sub.depth}
              ] as $edges
              | ([$all[]|select(.task.blob_id=="" and .task.depth==0)]) as $roots
              | if ($roots|length)!=1 then
                  error("expected exactly one root task with empty blob_id at depth zero")
                elif any($all[]; .task.blob_id=="" and .task.depth!=0) then
                  error("empty blob_id is only valid for the depth-zero root")
                elif any($edges[]; bad_edge(.; $all; $root; $chunkmap)) then
                  error("invalid parent-child task edge")
                elif ($edges|length)!=(($all|length)-1) then
                  error("task graph is disconnected or has duplicate edge records")
                elif (($edges|map(.child)|unique|length)!=($edges|length)) then
                  error("child task has multiple parents")
                elif any($all[]; . as $tr | $tr.task.blob_id!="" and ([$edges[] | select(.child==$tr.task.path)]|length)!=1) then
                  error("non-root task is missing one incoming edge")
                else
                  {
                    schema_version:"symthaea.unblob-semantic-projection.v1",
                    tasks: (
                      [ $all[] as $tr
                        | {
                            path:relpath($root; $tr.task.path),
                            depth:$tr.task.depth,
                            is_multi_file:$tr.task.is_multi_file,
                            via_producer:(if $tr.task.blob_id=="" then null else producer_key_global($chunkmap; $tr.task.blob_id) end),
                            reports:([$tr.reports[]|canonical_report($root)]|sort_by(stablejson)),
                            children:(
                              [ $tr.subtasks[]? as $sub
                                | {path:relpath($root; $sub.path), depth:$sub.depth,
                                   is_multi_file:$sub.is_multi_file,
                                   edge_kind:edge_kind($tr; $sub),
                                   via_producer:(if $sub.blob_id==$tr.task.blob_id and $sub.depth==$tr.task.depth
                                     then producer_key_global($chunkmap; $sub.blob_id)
                                     else producer_key($chunkmap; relpath($root; $tr.task.path); $sub.blob_id) end)}
                              ]|sort_by(.path)
                            )
                          }
                      ]|sort_by(.path)
                    )
                  }
                end
            end
        end
    end;

project($root)
