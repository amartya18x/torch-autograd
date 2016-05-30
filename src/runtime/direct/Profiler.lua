local util = require 'autograd.util'
local ffi = require 'ffi'

local Profiler = { }
Profiler.__index = Profiler

ffi.cdef([[
typedef long time_t;
typedef struct timeval {
 time_t tv_sec;
 time_t tv_usec;
};
struct rusage {
             struct timeval ru_utime; /* user time used */
             struct timeval ru_stime; /* system time used */
             long ru_maxrss;          /* integral max resident set size */
             long ru_ixrss;           /* integral shared text memory size */
             long ru_idrss;           /* integral unshared data size */
             long ru_isrss;           /* integral unshared stack size */
             long ru_minflt;          /* page reclaims */
             long ru_majflt;          /* page faults */
             long ru_nswap;           /* swaps */
             long ru_inblock;         /* block input operations */
             long ru_oublock;         /* block output operations */
             long ru_msgsnd;          /* messages sent */
             long ru_msgrcv;          /* messages received */
             long ru_nsignals;        /* signals received */
             long ru_nvcsw;           /* voluntary context switches */
             long ru_nivcsw;          /* involuntary context switches */
};
int getrusage(int who, struct rusage *r_usage);
]])

Profiler.time = ffi.new("struct rusage")


function Profiler.new()
   local p = { }
   p.lineMap = { }
   p.entries = { }
   p.times = 0
   setmetatable(p, Profiler)
   return p
end

function Profiler:mark(fun, level)
   local name = fun.name
   if fun.raw then
      name = fun.raw.__typename
      if name == nil or name == "" then
         name = "(nn object)"
      end
   end
   local di = debug.getinfo(level + 1)
   local line = di.short_src .. ":" .. di.currentline
   local fnMap = self.lineMap[line]
   if fnMap == nil then
      fnMap = { }
      self.lineMap[line] = fnMap
   end
   local entryIndex = fnMap[name]
   if entryIndex == nil then
      entryIndex = #self.entries + 1
      self.entries[entryIndex] = {
         debuginfo = di,
         name = name,
         line = line,
         forwardTime = 0,
         backwardTime = 0,
         forwardMem = 0,
         backwardMem = 0
      }
      fnMap[name] = entryIndex
   end
   return entryIndex
end

function Profiler:markCycle()
   self.times = self.times + 1
end

function Profiler:measureForward(id, time, mem)
   self.entries[id].forwardTime = self.entries[id].forwardTime + time
   self.entries[id].forwardMem = self.entries[id].forwardMem + mem
end

function Profiler:measureBackward(id, time, mem)
   self.entries[id].backwardTime = self.entries[id].backwardTime + time
   self.entries[id].backwardMem = self.entries[id].backwardMem + mem
end

function pctStr(n, tot)
   return tostring(math.floor((n / tot) * 100.0)) .. "%"
end

function padMin(s, min)
   if #s < min then
      return s .. string.rep(" ", min - #s)
   end
   return s
end

function Profiler:printReport(type)
   local totalForward = 0
   local totalBackward = 0
   local totalFMem = 0
   local totalBMem = 0
   for i = 1, #self.entries do
      local t = self.entries[i]
      totalForward = totalForward + t.forwardTime
      totalBackward = totalBackward + t.backwardTime
      totalFMem = totalFMem + t.forwardMem
      totalBMem = totalBMem + t.backwardMem
   end

   local timeSorted = util.shallowCopy(self.entries)
   table.sort(timeSorted, function(a, b)
      return (a.forwardTime + a.backwardTime) > (b.forwardTime + b.backwardTime)
   end)
   print("")
   print(string.format("[autograd] average forward time: %.2fms", (totalForward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average backward time: %.2fms", (totalBackward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average overall time: %.2fms", ((totalForward + totalBackward) / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average forward time: %.2fms", (totalForward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average backward time: %.2fms", (totalBackward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average overall time: %.2fms", ((totalForward + totalBackward) / (self.times + 1)) * 1000.0))
   print("[autograd] top operations:")
   if type == "detailed" then
      print("[autograd] " .. string.rep("=", 80))
      print("[autograd] " .. padMin("name", 20), "fwd", "bwd", "ovr", "line")
      print("[autograd] " .. string.rep("=", 80))
      for i = 1, math.min(10, #timeSorted) do
         local t = timeSorted[i]
         print("[autograd] " .. padMin(t.name, 20), pctStr(t.forwardTime, totalForward), pctStr(t.backwardTime, totalBackward), pctStr(t.forwardTime + t.backwardTime, totalForward + totalBackward), t.line)
      end
   end

   
   local MemSorted = util.shallowCopy(self.entries)
   table.sort(MemSorted, function(a, b)
      return (a.forwardMem + a.backwardMem) > (b.forwardMem + b.backwardMem)
   end)
   print("")
   print(string.format("[autograd] average forward Memory: %.2f mb", (totalFMem / (self.times + 1)/1024) ))
   print(string.format("[autograd] average backward Memory: %.2f mb", (totalBMem / (self.times + 1)/1024) ))
   print(string.format("[autograd] average overall Memory: %.2f mb", ((totalFMem + totalBMem) / (self.times + 1)/1024) ))
   print(string.format("[autograd] average forward Memory: %.2f mb", (totalFMem / (self.times + 1)/1024)))
   print(string.format("[autograd] average backward Memory: %.2f mb", (totalBMem / (self.times + 1)/1024)))
   print(string.format("[autograd] average overall Memory: %.2f mb", ((totalFMem + totalBMem) / (self.times + 1) / 1024) ))
   print("[autograd] top operations:")
   if type == "detailed" then
      print("[autograd] " .. string.rep("=", 80))
      print("[autograd] " .. padMin("name", 20), "fwd", "bwd", "ovr", "line")
      print("[autograd] " .. string.rep("=", 80))
      for i = 1, math.min(10, #MemSorted) do
         local t = MemSorted[i]
         print("[autograd] " .. padMin(t.name, 20), pctStr(t.forwardMem, totalFMem), pctStr(t.backwardMem, totalBMem), pctStr(t.forwardMem + t.backwardMem, totalFMem + totalBMem), t.line)
      end
   end

   
   print("")
end

return Profiler
