# ENG-REA-005: canonical semantic projection of an Unblob ProcessResult JSON report.
# --arg root ABSOLUTE_ROOT names the disposable extraction root for this run.
# Deliberately fails closed on unknown report types / dangling or ambiguous lineage.

def uint($n): ($n|type)=="number" and $n>=0 and ($n|floor)==$n;

def stablejson: walk(if type=="object" then (to_entries|sort_by(.key)|from_entries) else . end)|tojson;

def known_report_types: [
  "UnknownError", "CalculateChunkExceptionReport", "CalculateMultiFileExceptionReport",
  "ExtractCommandFailedReport", "OutputDirectoryExistsReport", "ExtractorDependencyNotFoundReport",
  "ExtractedFileDeletedReport", "ExtractorTimedOut", "MaliciousSymlinkRemoved",
  "MultiFileCollisionReport", "StatReport", "HashReport", "FileMagicReport",
  "RandomnessReport", "EncryptionMetadataReport", "ChunkReport", "UnknownChunkReport",
  "CarveDirectoryReport", "MultiFileReport", "ExtractionProblem", "PathTraversalProblem",
  "LinkExtractionProblem", "SpecialFileExtractionProblem"
];

def relpath($root; $p):
  if ($p|type)!="string" then error("path field is not a string")
  elif $p==$root then "."
  elif ($p|startswith($root + "/")) then $p[(($root|length)+1):]
  elif ($p|startswith("/")) then error("absolute path escapes declared extraction root: " + $p)
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

def bad_task_result:
  if type!="object" then true
  elif (keys|sort)!=["reports", "subtasks", "task"] then true
  elif (.task|type)!="object" then true
  elif ((.task|keys|sort)!=["chunk_id", "depth", "path"]) then true
  elif (.task.path|type)!="string" or (.task.path|length)==0 then true
  elif (uint(.task.depth)|not) then true
  elif (.task.chunk_id|type)!="string" then true
  elif (.reports|type)!="array" then true
  elif (.subtasks|type)!="array" then true
  elif any(.subtasks[]; type!="object" or ((keys|sort)!=["chunk_id", "depth", "path"]) or
                           (.path|type)!="string" or (.path|length)==0 or
                           (uint(.depth)|not) or (.chunk_id|type)!="string" or (.chunk_id|length)==0) then true
  else false end;

def bad_chunk($task; $chunk):
  if ($chunk.id|type)!="string" or ($chunk.id|length)==0 then true
  elif (uint($chunk.start_offset)|not) or (uint($chunk.end_offset)|not) or (uint($chunk.size)|not) then true
  elif $chunk.end_offset < $chunk.start_offset then true
  elif $chunk.size != ($chunk.end_offset-$chunk.start_offset) then true
  else
    ([$task.reports[]? | select(.__typename__=="StatReport") | .size] | first) as $task_size
    | if $task_size==null then false
      elif (uint($task_size)|not) then true
      else $chunk.end_offset > $task_size end
  end;

def chunk_key($chunkmap; $id):
  ($chunkmap|map(select(.id==$id))) as $matches
  | if ($matches|length)!=1 then error("missing or ambiguous chunk reference: " + $id)
    else $matches[0].key end;

def local_chunk_key($root; $chunkmap; $parent_path; $id):
  ($chunkmap|map(select(.id==$id and .owner==(relpath($root; $parent_path)))) ) as $matches
  | if ($matches|length)!=1 then error("subtask does not reference one local parent chunk: " + $id)
    else $matches[0].key end;

def bad_edge($edge; $all; $root; $chunkmap):
  ([$all[]|select(.task.path==$edge.parent)] ) as $parents
  | ([$all[]|select(.task.path==$edge.child)] ) as $children
  | if ($parents|length)!=1 or ($children|length)!=1 then true
    elif $edge.parent==$edge.child then true
    elif ($edge.chunk_id|type)!="string" or ($edge.chunk_id|length)==0 then true
    elif $edge.depth != $parents[0].task.depth+1 then true
    elif $edge.depth != $children[0].task.depth then true
    elif $edge.chunk_id != $children[0].task.chunk_id then true
    elif ([$parents[0].reports[]?|select((.__typename__=="ChunkReport" or .__typename__=="MultiFileReport") and .id==$edge.chunk_id)]|length)!=1 then true
    else
      # Resolve here too so dangling / malformed keys cannot be hidden by projection.
      (local_chunk_key($root; $chunkmap; $edge.parent; $edge.chunk_id) | type)!="object"
    end;

def project($root):
  . as $all
  | if (($root|type)!="string") or ((($root|startswith("/"))|not)) or ($root|endswith("/")) then
      error("root must be an absolute path without a trailing slash")
    elif ($all|type)!="array" or ($all|length)==0 then
      error("provider report must be a nonempty TaskResult array")
    elif any($all[]; bad_task_result) then
      error("malformed TaskResult / Task / subtask schema")
    else
      # Normalization also validates all root and descendant report types and paths.
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
          | if any($all[]; . as $tr | any($tr.reports[]?; ((.__typename__=="ChunkReport") and (bad_chunk($tr; .))))) then
              error("malformed chunk interval or out-of-bounds chunk")
            elif (($chunkmap|map(.id)|unique|length)!=($chunkmap|length)) then
              error("duplicate provider chunk ID")
            elif any($chunkmap[]; (.id|type)!="string" or (.id|length)==0) then
              error("empty provider chunk ID")
            else
              [ $all[] as $parent | $parent.subtasks[]? as $sub
                | {parent:$parent.task.path, child:$sub.path, chunk_id:$sub.chunk_id, depth:$sub.depth}
              ] as $edges
              | ([$all[]|select(.task.chunk_id=="")]) as $roots
              | if ($roots|length)!=1 or $roots[0].task.depth!=0 then
                  error("expected exactly one depth-zero root task")
                elif any($all[]; .task.chunk_id!="" and (.task.chunk_id as $id | ($chunkmap|map(select(.id==$id))|length)!=1)) then
                  error("descendant task references missing / ambiguous parent chunk")
                elif any($edges[]; bad_edge(.; $all; $root; $chunkmap)) then
                  error("invalid parent-child task edge")
                elif ($edges|length)!=(($all|length)-1) then
                  error("task graph is disconnected or has duplicate edge records")
                elif (($edges|map(.child)|unique|length)!=($edges|length)) then
                  error("child task has multiple parents")
                else
                  {
                    schema_version:"symthaea.unblob-semantic-projection.v1",
                    tasks: (
                      [ $all[] as $tr
                        | {
                            path:relpath($root; $tr.task.path),
                            depth:$tr.task.depth,
                            via_chunk:(if $tr.task.chunk_id=="" then null else chunk_key($chunkmap; $tr.task.chunk_id) end),
                            reports:([$tr.reports[]|canonical_report($root)]|sort_by(stablejson)),
                            children:(
                              [ $tr.subtasks[]? as $sub
                                | {path:relpath($root; $sub.path), depth:$sub.depth,
                                   via_chunk:local_chunk_key($root; $chunkmap; $tr.task.path; $sub.chunk_id)}
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
