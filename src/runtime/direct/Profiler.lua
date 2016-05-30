local util = require 'autograd.util'

local Profiler = { }
Profiler.__index = Profiler

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
   print(string.format("[autograd] average forward Memory: %.2fms", (totalFMem / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average backward Memory: %.2fms", (totalBMem / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average overall Memory: %.2fms", ((totalFMem + totalBMem) / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average forward Memory: %.2fms", (totalFMem / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average backward Memory: %.2fms", (totalBMem / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average overall Memory: %.2fms", ((totalFMem + totalBMem) / (self.times + 1)) * 1000.0))
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
